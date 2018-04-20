/*
 * Copyright 2018 org.LTR4L
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ltr4l.trainers;

import org.ltr4l.boosting.Ensemble;
import org.ltr4l.boosting.RegressionTree;
import org.ltr4l.boosting.RegressionTree.Split;
import org.ltr4l.boosting.TreeEnsemble;
import org.ltr4l.boosting.TreeTools;
import org.ltr4l.nn.Activation;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.DataProcessor;
import org.ltr4l.tools.Error;

import java.io.IOException;
import java.io.Reader;
import java.util.*;
import java.util.stream.Collectors;

import static org.ltr4l.boosting.TreeTools.findMinLossFeat;
import static org.ltr4l.boosting.TreeTools.orderByFeature;

public class LambdaMartTrainer extends AbstractTrainer<Ensemble, Ensemble.TreeConfig> {
  private final List<Document> trainingDocs;
  private final List<Document> validationDocs;
  private final List<Document[][]> trainingPairs;
  private final List<List<Document>> featureSortedDocs;
  private final double[][] thresholds;
  private final int numTrees;
  private final int numLeaves;
  private final double lrRate;

  private static final double DEFAULT_VARIANCE_TOLERANCE = 0;

  LambdaMartTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override);
    trainingDocs = DataProcessor.makeDocList(trainingSet);
    validationDocs = DataProcessor.makeDocList(validationSet);
    trainingPairs = trainingSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
    numTrees = config.getNumTrees();
    numLeaves = config.getNumLeaves();
    lrRate = config.getLearningRate();

    featureSortedDocs = new ArrayList<>();
    //{
    // {threshold, calculateScore}, //Feature 0
    // {threshold, calculateScore}, //Feature 1
    // ...
    //}
    thresholds = new double[training.getFeatureLength()][2];
    for (int feat = 0; feat < training.getFeatureLength(); feat++) {
      featureSortedDocs.add(orderByFeature(trainingDocs, feat));
      thresholds[feat] = TreeTools.findThreshold(featureSortedDocs.get(feat), feat);
    }
  }



  @Override
  protected Ensemble constructRanker() {
    return new Ensemble();
  }

  @Override
  double calculateLoss(List<Query> queries) { //TODO: Implement
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> errorFunc.error(ranker.predict(doc.getFeatures()), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  @Override
  protected Error makeErrorFunc() {
    return new Error.Square();
  }

  @Override
  public void trainAndValidate() {
    train();
    validate(numTrees, evalK);
    report.close();
/*    try {
      if(!config.nomodel)
        ranker.writeModel(config, modelFile);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }*/
  }

  @Override
  public void train() {

    Map<Document, Double> pws = new HashMap<>();
    double minLoss = 0;
    for(Document doc : trainingDocs) pws.put(doc,Math.pow(2, doc.getLabel()) - 1 );
    //trainingDocs.forEach(doc -> pws.put(doc, pws.put(doc, Math.pow(2, doc.getLabel()) - 1)));

    HashMap<Document, Double> ranks = new HashMap<>(); //TODO: More efficient way?
    HashMap<Document, Double> lambdas = new HashMap<>();
    HashMap<Document, Double> logs = new HashMap<>();
    HashMap<Document, Double> lambdaDers = new HashMap<>();

    int minLossFeat = findMinLossFeat(thresholds, minLoss);

    for (int t = 1; t <= numTrees; t++){
      //First, calculate lambdas for this iteration.
      for (int iq = 0; iq < trainingSet.size(); iq++) {
        if (trainingPairs.get(iq) == null) //As we are skipping these, they must not influence leaf scores.
          continue;

        Query query = trainingSet.get(iq);
        double N = LambdaRankTrainer.idcg(query.getDocList(), query.getDocList().size());
        List<Document> sorted = ranker.sort(query);
        for (int i = 0; i < sorted.size(); i++) {
          Document doc = sorted.get(i);
          ranks.put(doc, ranker.predict(doc.getFeatures()));
          lambdas.put(doc, 0d);
          lambdaDers.put(doc, 0d);
          logs.put(doc, 1 / Math.log(i + 2));
        }

        for (Document[] pair : trainingPairs.get(iq)) {
          double dNCG = (pws.get(pair[0]) - pws.get(pair[1])) * (logs.get(pair[0]) - logs.get(pair[1])) / N;
          double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj) ; sigmoid has minus sign
          double lambda = Math.abs(new Activation.Sigmoid().output(diff) * dNCG); //TODO: Make static method or class variable
          double lambdaDer = lambda * (1 - Math.abs(new Activation.Sigmoid().output(diff)));
          lambdas.put(pair[0], lambdas.get(pair[0]) - lambda); //λ1 = λ1 - dλ
          lambdas.put(pair[1], lambdas.get(pair[1]) + lambda); //λ2 = λ2 - dλ
          lambdaDers.put(pair[0], lambdaDers.get(pair[0]) - lambdaDer);
          lambdaDers.put(pair[1], lambdaDers.get(pair[1]) + lambdaDer);
        }
      }
      //Then create tree
      double[] minThresholdLoss = thresholds[minLossFeat];
      double minThreshold = minThresholdLoss[0];
      RegressionTree tree = new RegressionTree(numLeaves, minLossFeat, minThreshold, trainingDocs);
      tree.setWeight(lrRate);
      ranker.addTree(tree);
      minLoss = minThresholdLoss[1]; //For the next tree.

      List<Split> terminalLeaves = tree.getTerminalLeaves();
      //Assign lambdas as leaf scores
      for(Split leaf : terminalLeaves){
        double y = leaf.getScoredDocs().stream().filter(doc -> lambdas.containsKey(doc)).mapToDouble(doc -> lambdas.get(doc)).sum();
        double w = leaf.getScoredDocs().stream().filter(doc -> lambdas.containsKey(doc)).mapToDouble(doc -> lambdaDers.get(doc)).sum();
        if(w == 0) w += 1e-8; //To avoid dividing by zero
        leaf.setScore(y / w);
      }
      System.out.printf("Tree number %d completed \n", t);
      validate(t, evalK);
    }
  }

  @Override
  public Class<Ensemble.TreeConfig> getConfigClass() {
    return getCC();
  }

  public static Class<Ensemble.TreeConfig> getCC(){
    return Ensemble.TreeConfig.class;
  }

}
