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

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.ltr4l.nn.Activation;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.RankNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.*;
import org.ltr4l.tools.Error;

/**
 * The implementation of MLPTrainer which uses the
 * RankNet algorithm.
 *
 * C.J.C. Burges, T. Shaked, E. Renshaw, A. Lazier, M. Deeds, N. Hamilton, G. Hullender: Learning
 * to Rank using Gradient Descent . Proceedings of the International Conference on Machine
 * Learning, 2005.
 *
 */
public class RankNetTrainer extends MLPTrainer<RankNetMLP> {
  protected final List<Document[][]> trainingPairs;
  protected final List<Document[][]> validationPairs;

  RankNetTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override, true);

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
  protected RankNetMLP constructRanker(){
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new RankNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Entropy();
  }

  @Override
  protected LossCalculator makeLossCalculator(){
    return new PairwiseLossCalc.RankNetLossCalc<>(ranker, trainingSet, validationSet, errorFunc);
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
        double s1 = ranker.forwardProp(pair[0]);
        double s2 = ranker.forwardProp(pair[1]);
        double output = Math.pow(1 + Math.exp(s2 - s1), -1); //double output = new Activation.Sigmoid().output(s1 - s2);
        queryLoss += errorFunc.error(output, 1d);
        //queryLoss += Math.log(1 + Math.exp(s2 - s1)); This is a derivation; equivalent to errorFunc.error.
      }
      loss += queryLoss / query.length;
    }
    return loss / processedQueryNum;
  }

  @Override
  public void train() {
    double threshold = 0.5;
    //Present all docs of randomly selected query
    //For number of queries / 6 times.
    int numTrained = 0;
    for (int i = 0; i < trainingPairs.size() / 6; i++) {
      int iq = new Random().nextInt(trainingPairs.size());
      if (trainingPairs.get(iq) == null)
        //i--;  //Note: if all queries have null for document pairs, will loop infinitely.
        continue;

      for (Document[] docPair : trainingPairs.get(iq)) { //for each document pair in query iq
        Document docA = docPair[0];
        Document docB = docPair[1];

        double si = ranker.forwardProp(docA);
        double sj = ranker.forwardProp(docB);
        double delta = si - sj;

        if (delta < threshold) {
          double sigma = new Activation.Sigmoid().output(-delta);
          ranker.backProp(sigma);
          ranker.forwardProp(docA);
          ranker.backProp(-sigma);
          numTrained++;
          if(batchSize == 0) ranker.updateWeights(lrRate, rgRate);
        }
      }
      if (batchSize != 0 && ((numTrained) % batchSize) == 0){
        ranker.updateWeights(lrRate, rgRate);
      }
    }
    if (batchSize != 0) ranker.updateWeights(lrRate, rgRate); //Update at the end of the epoch, regardless of batchSize.
  }
}

