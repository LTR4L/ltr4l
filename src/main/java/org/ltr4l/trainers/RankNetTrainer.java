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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.ltr4l.nn.Activation;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.RankNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Regularization;

public class RankNetTrainer extends MLPTrainer {
  protected RankNetMLP rmlp;
  protected List<Document[][]> trainingPairs;
  protected List<Document[][]> validationPairs;

  RankNetTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config, true);
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    networkShape.add(1, new Activation.Identity());
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    rmlp = new RankNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
    super.mlp = rmlp;


    trainingPairs = new ArrayList<>();
    for (int i = 0; i < trainingSet.size(); i++) {
      Query query = trainingSet.get(i);
      Document[][] documentPairs = query.orderDocPairs();
      trainingPairs.add(documentPairs);                   //add even if null, as placeholder for query.
    }

    validationPairs = new ArrayList<>();
    for (int i = 0; i < validationSet.size(); i++) {
      Query query = validationSet.get(i);
      Document[][] documentPairs = query.orderDocPairs();
      validationPairs.add(documentPairs);
    }
  }

  @Override
  public double calculateLoss(List<Query> queries) {
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
        double s1 = rmlp.forwardProp(pair[0]);
        double s2 = rmlp.forwardProp(pair[1]);
        //double output = Math.pow(1 + Math.exp(s2 - s1), -1);
        //double output = new Activation.Sigmoid().output(s1 - s2);
        //queryLoss += new Error.ENTROPY().error(output, 1d);
        queryLoss += Math.log(1 + Math.exp(s2 - s1));
      }
      loss += queryLoss / query.length;
    }
    return loss / processedQueryNum; //(double) (docPairs.size() - nullQueryNum);
  }

  @Override
  public void train() {
    double threshold = 0.5;

    //Present all docs of randomly selected query
    //For number of queries / 6 times.
    for (int i = 0; i < trainingPairs.size() / 6; i++) {
      int iq = new Random().nextInt(trainingPairs.size());
      if (trainingPairs.get(iq) == null) {
        //i--;  //Note: if all queries have null for document pairs, will loop infinitely.
        continue;
      }
      for (Document[] docPair : trainingPairs.get(iq)) { //for each document pair in query iq
        Document docA = docPair[0];
        Document docB = docPair[1];

        double si = rmlp.forwardProp(docA);
        double sj = rmlp.forwardProp(docB);
        double delta = si - sj;

        if (delta < threshold) {
          double sigma = new Activation.Sigmoid().output(-delta);
          rmlp.backProp(sigma);
          rmlp.forwardProp(docA);
          rmlp.backProp(-sigma);
          rmlp.updateWeights(lrRate, rgRate);
        }
      }
    }
  }
}

