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

import org.ltr4l.boosting.TreeEnsemble;
import org.ltr4l.nn.Activation;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.DataProcessor;
import org.ltr4l.tools.Error;

import java.io.Reader;
import java.util.*;
import java.util.stream.Collectors;

public class LambdaMartTrainer extends AbstractTrainer<TreeEnsemble, TreeEnsemble.TreeConfig> {
  private final List<Document> trainingDocs;
  private final List<Document> validationDocs;
  private final List<Document[][]> trainingPairs;
  private final Map<Integer, Double> variance;
  private final List<Integer> selectedFeatures;
  private static final double DEFAULT_VARIANCE_TOLERANCE = 0;

  LambdaMartTrainer(QuerySet training, QuerySet validation, Reader reader, Config override){
    super(training, validation, reader, override);
    trainingDocs = DataProcessor.makeDocList(trainingSet);
    validationDocs = DataProcessor.makeDocList(validationSet);
    trainingPairs = trainingSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
    List<Document> dataSet = DataProcessor.makeDocList(trainingSet);
    variance = DataProcessor.calcVariances(dataSet);
    selectedFeatures = DataProcessor.orderSelectedFeatures(variance, DEFAULT_VARIANCE_TOLERANCE);
  }

  @Override
  protected TreeEnsemble constructRanker() {
    List<Document> dataSet = DataProcessor.makeDocList(trainingSet);
    Map<Integer, Double> variances = DataProcessor.calcVariances(dataSet);
    List<Integer> chosenFeatures = DataProcessor.orderSelectedFeatures(variances, DEFAULT_VARIANCE_TOLERANCE);
    return new TreeEnsemble(config, chosenFeatures.get(0));
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
  public void train() {
    for (int iq = 0; iq < trainingSet.size(); iq++) {
      if (trainingPairs.get(iq) == null)
        continue;

      Query query = trainingSet.get(iq);

      HashMap<Document, Double> ranks = new HashMap<>(); //TODO: More efficient way?
      HashMap<Document, Double> lambdas = new HashMap<>();
      HashMap<Document, Double> pws = new HashMap<>();
      HashMap<Document, Double> logs = new HashMap<>();
      HashMap<Document, Double> lambdaDers = new HashMap<>();
      double N = LambdaRankTrainer.idcg(query.getDocList(), query.getDocList().size());


      List<Document> sorted = ranker.sort(query);

      for (int i = 0; i < sorted.size(); i++) {
        Document doc = sorted.get(i);
        ranks.put(doc, ranker.predict(doc.getFeatures()));
        lambdas.put(doc, 0d);
        lambdaDers.put(doc, 0d);
        pws.put(doc, Math.pow(2, doc.getLabel()) - 1);
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

      //TODO: Add tree
    }
  }

  @Override
  public Class<TreeEnsemble.TreeConfig> getConfigClass() {
    return getCC();
  }

  public static Class<TreeEnsemble.TreeConfig> getCC(){
    return TreeEnsemble.TreeConfig.class;
  }

}
