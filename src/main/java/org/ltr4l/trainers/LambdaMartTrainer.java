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

import org.ltr4l.boosting.Tree;
import org.ltr4l.boosting.TreeEnsemble;
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

public class LambdaMartTrainer extends AbstractTrainer<TreeEnsemble, TreeEnsemble.TreeConfig> {
  private final List<Document> trainingDocs;
  private final List<Document> validationDocs;
  private final List<Document[][]> trainingPairs;
  private final Document[][] featureSortedDocs;
  private final double[][] thresholds;
  private final int numTrees;
  private final int numLeaves;

  private static final double DEFAULT_VARIANCE_TOLERANCE = 0;

  LambdaMartTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override);
    trainingDocs = DataProcessor.makeDocList(trainingSet);
    validationDocs = DataProcessor.makeDocList(validationSet);
    trainingPairs = trainingSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());

    numTrees = config.getNumTrees();
    numLeaves = config.getNumLeaves();

    featureSortedDocs = new Document[training.getFeatureLength()][trainingDocs.size()];
    //{
    // {threshold, score}, //Feature 0
    // {threshold, score}, //Feature 1
    // ...
    //}
    thresholds = new double[training.getFeatureLength()][2];
    for (int feat = 0; feat < training.getFeatureLength(); feat++) {
      featureSortedDocs[feat] = orderByFeature(trainingDocs, feat);
      thresholds[feat] = findThreshold(featureSortedDocs[feat], feat);
    }

  }



  @Override
  protected TreeEnsemble constructRanker() {
    return new TreeEnsemble();
  }

  @Override
  double calculateLoss(List<Query> queries) { //TODO: Implement
    return 0;
  }

  @Override
  protected Error makeErrorFunc() {
    return new Error.Square();
  }

  @Override
  public void trainAndValidate(){
    for (int t = 1; t <= numTrees; t++) {
      train();
      validate(t, evalK);
    }
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
    trainingDocs.forEach(doc -> pws.put(doc, pws.put(doc, Math.pow(2, doc.getLabel()) - 1)));

    HashMap<Document, Double> ranks = new HashMap<>(); //TODO: More efficient way?
    HashMap<Document, Double> lambdas = new HashMap<>();
    HashMap<Document, Double> logs = new HashMap<>();
    HashMap<Document, Double> lambdaDers = new HashMap<>();

    for (int t = 1; t <= numTrees; t++){
      for (int iq = 0; iq < trainingSet.size(); iq++) {
        if (trainingPairs.get(iq) == null)
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
          double dNCG = N * (pws.get(pair[0]) - pws.get(pair[1])) * (logs.get(pair[0]) - logs.get(pair[1]));
          double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj)
          double lambda = Math.abs(new Activation.Sigmoid().output(diff) * dNCG); //TODO: Make static method or class variable
          double lambdaDer = lambda * (1 - Math.abs(new Activation.Sigmoid().output(diff)));
          lambdas.put(pair[0], lambdas.get(pair[0]) - lambda); //λ1 = λ1 - dλ
          lambdas.put(pair[1], lambdas.get(pair[1]) + lambda); //λ2 = λ2 - dλ
          lambdaDers.put(pair[0], lambdaDers.get(pair[0]) - lambdaDer);
          lambdaDers.put(pair[1], lambdaDers.get(pair[1]) + lambdaDer);
        }



      }
    }
  }

  //Build tree using all of the data. The initial feature will determine which feature
  //from thresholds to use for the initialization.
  protected Tree makeTree(int feature){
    //Make root
    double threshold = thresholds[feature][0];
    Tree tree = new Tree(feature, threshold);
    for(int lId = 0; lId < numLeaves - 2; lId++){ //Add leaves
      Document[] lDocs = Arrays.stream(featureSortedDocs[feature])
          .filter(doc -> doc.getFeature(feature) <= threshold)
          .toArray(Document[]::new);
      Document[] rDocs = Arrays.stream(featureSortedDocs[feature])
          .filter(doc -> doc.getFeature(feature) > threshold)
          .toArray(Document[]::new);


    }
  }


  protected static double[] findThreshold(Document[] fSortedDocs, int feat){
    int numDocs = fSortedDocs.length;
    double threshold = fSortedDocs[0].getFeature(feat);
    double minLoss = Double.POSITIVE_INFINITY;
    for(int threshId = 0; threshId < numDocs; threshId++ ){
      //Consider cases where feature values are the same!
      if (fSortedDocs[threshId].getFeature(feat) == fSortedDocs[threshId + 1].getFeature(feat)) continue;
      Document[] lDocs = Arrays.copyOfRange(fSortedDocs, 0, threshId + 1);
      Document[] rDocs = Arrays.copyOfRange(fSortedDocs, threshId, numDocs + 1);
      double loss = calcThresholdLoss(lDocs) + calcThresholdLoss(rDocs);
      if(loss < minLoss){
        threshold = threshId;
        minLoss = loss;
      }
    }
    return new double[] {threshold, minLoss};
  }

  protected static double calcThresholdLoss(Document[] subData){
    if (subData.length == 0) return 0;
    double avg = Arrays.stream(subData).mapToDouble(doc -> doc.getLabel()).sum() / subData.length;
    return Arrays.stream(subData).mapToDouble(doc -> Math.pow(doc.getLabel() - avg, 2)).sum();
  }

  protected static int findMinLossFeat(double[][] thresholds){
    return findMinLossFeat(thresholds, 0.0d);
  }

  protected static int findMinLossFeat(double[][] thresholds, double minLoss){
    int feat = 0;
    double loss = Double.POSITIVE_INFINITY;
    for (int fid  = 0; feat < thresholds.length; feat++){
      if(thresholds[feat][1] < loss && thresholds[feat][1] > minLoss){
        loss = thresholds[feat][1];
        feat = fid;
      }
    }
    return feat;
  }

  @Override
  public Class<TreeEnsemble.TreeConfig> getConfigClass() {
    return getCC();
  }

  public static Class<TreeEnsemble.TreeConfig> getCC(){
    return TreeEnsemble.TreeConfig.class;
  }

  protected static Document[] orderByFeature(List<Document> documents, int feature){
    return documents.stream().sorted(Comparator.comparingDouble(doc -> doc.getFeature(feature))).toArray(Document[]::new);
  }

}
