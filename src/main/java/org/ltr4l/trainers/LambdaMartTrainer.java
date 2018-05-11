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

import org.ltr4l.boosting.*;
import org.ltr4l.boosting.RegressionTree.Split;
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

public class LambdaMartTrainer extends AbstractTrainer<Ensemble, Ensemble.TreeConfig> {
  private final List<Document> trainingDocs;
  private final List<Document[][]> trainingPairs;
  private final List<Document[][]> validationPairs;
  private final List<FeatureSortedDocs> featureSortedDocs;
  private final double[][] thresholds;
  private final int numTrees;
  private final int numLeaves;
  private final double lrRate;

  private static final double DEFAULT_VARIANCE_TOLERANCE = 0;

  LambdaMartTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override);
    trainingDocs = DataProcessor.makeDocList(trainingSet);
    trainingPairs = trainingSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
    validationPairs = validationSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
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
      featureSortedDocs.add(FeatureSortedDocs.get(trainingDocs, feat));
      thresholds[feat] = TreeTools.findThreshold(featureSortedDocs.get(feat));
    }
  }



  @Override
  protected Ensemble constructRanker() {
    return new Ensemble();
  }

  @Override
  double calculateLoss(List<Query> queries) { //TODO: Implement
    List<Document[][]> docPairs;
    if (queries == trainingSet)
      docPairs = trainingPairs;
    else if (queries == validationSet)
      docPairs = validationPairs;
    else
      return -1d;
    double loss = 0d;
    int processedQueryNum = 0;
    for (Document[][] query : docPairs) {
      if (query == null)
        continue;
      processedQueryNum++;
      double queryLoss = 0d;
      for (Document[] pair : query) {
        double s1 = ranker.predict(pair[0].getFeatures());
        double s2 = ranker.predict(pair[1].getFeatures());
        double output = Math.pow(1 + Math.exp(s2 - s1), -1); //double output = new Activation.Sigmoid().output(s1 - s2);
        queryLoss += errorFunc.error(output, 1d);
      }
      loss += queryLoss / query.length;
    }
    return loss / processedQueryNum;
  }

  @Override
  protected Error makeErrorFunc() {
    return new Error.Entropy();
  }

  @Override
  public void trainAndValidate() {
    train();
    //validate(numTrees, evalK);
    report.close();
    try {
      if(!config.nomodel)
        ranker.writeModel(config, modelFile);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
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

    for (int t = 1; t <= numTrees; t++){
      int minLossFeat = findMinLossFeat(thresholds, minLoss);
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
          double lambdaDer = lambda * (1 - (lambda/dNCG));
          lambdas.put(pair[0], lambdas.get(pair[0]) - lambda); //λ1 = λ1 - dλ
          lambdas.put(pair[1], lambdas.get(pair[1]) + lambda); //λ2 = λ2 - dλ
          lambdaDers.put(pair[0], lambdaDers.get(pair[0]) - lambdaDer);
          lambdaDers.put(pair[1], lambdaDers.get(pair[1]) + lambdaDer);
        }
      }
      //Then create tree
      double[] minThresholdLoss = thresholds[minLossFeat];
      double minThreshold = minThresholdLoss[0];
      RegressionTree tree;
      try {
        tree = new RegressionTree(numLeaves, minLossFeat, minThreshold, trainingDocs);
      }
      catch (InvalidFeatureThresholdException ie) {
        System.err.printf("Valid tree could not be created. Stopping training early at tree %d \n", t - 1);
        return; //TODO: Implement solution to continue creating trees. For now, stop training.
      }
      //RegressionTree tree = new RegressionTree(numLeaves, minLossFeat, minThreshold, trainingDocs);
      ranker.addTree(tree);
      minLoss = minThresholdLoss[1]; //For the next tree.

      List<Split> terminalLeaves = tree.getTerminalLeaves();
      //Assign lambdas as leaf scores
      for(Split leaf : terminalLeaves){
        double y = leaf.getScoredDocs().stream().filter(doc -> lambdas.containsKey(doc)).mapToDouble(doc -> lambdas.get(doc)).sum();
        double w = leaf.getScoredDocs().stream().filter(doc -> lambdaDers.containsKey(doc)).mapToDouble(doc -> lambdaDers.get(doc)).sum();
        //if(w == 0) w += 1e-8; //To avoid dividing by zero
        leaf.setScore(lrRate * y / w);
      }
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
