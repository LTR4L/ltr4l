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

import java.util.HashMap;

import org.ltr4l.nn.Activation;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

/**
 * An extension of RankNetTrainer.
 * FRank can be considered a "midway step" between RankNet
 * and LambdaRank.
 *
 */
public class FRankTrainer extends RankNetTrainer {


  FRankTrainer(QuerySet training, QuerySet validation, String config) {
    super(training, validation, config);
  }

  @Override
  public void train() {
    int numTrained = 0;
    for (int iq = 0; iq < trainingSet.size(); iq++) {  //index query
      if (trainingPairs.get(iq) == null)
        continue; //if there are no valid pairs for the query, skip.

      Query query = trainingSet.get(iq);
      //int qsize = query.getDocList().size();
      //double[] lambdas = new double[qsize];
      //double[] ranks = new double[qsize];
      HashMap<Document, Double> lambdas = new HashMap<>(); //lambdas
      HashMap<Document, Double> ranks = new HashMap<>();   //Create map for documents ranks.
      for (Document doc : query.getDocList()) {
        lambdas.put(doc, 0d);
        ranks.put(doc, ranker.forwardProp(doc));
      }

      for (Document[] pair : trainingPairs.get(iq)) {
        double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj)
        double lambda = new Activation.Sigmoid().output(diff);
        lambdas.put(pair[0], lambdas.get(pair[0]) - lambda); //λ1 = λ1 - dλ
        lambdas.put(pair[1], lambdas.get(pair[1]) + lambda); //λ2 = λ2 - dλ
      }

      for (Document doc : query.getDocList()) {
        ranker.forwardProp(doc);
        ranker.backProp(lambdas.get(doc));
        numTrained++;
        if (batchSize != 0 && numTrained % batchSize == 0) ranker.updateWeights(lrRate, rgRate);
      }
    }
    ranker.updateWeights(lrRate, rgRate); //Update at the end of the epoch, regardless of batchSize.
  }

}
