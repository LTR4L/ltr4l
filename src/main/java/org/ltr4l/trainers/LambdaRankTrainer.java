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

import org.ltr4l.nn.Activation;
import org.ltr4l.tools.Config;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

/**
 * LambdaRankTrainer trains the RankNetTrainer's network
 * through a different algorithm, which incorporates
 * ΔNDCG.
 * */
public class LambdaRankTrainer extends RankNetTrainer {

  LambdaRankTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config);
  }

  @Override
  public void train() {
    int numTrained = 0;
    for (int iq = 0; iq < trainingSet.size(); iq++) {
      if (trainingPairs.get(iq) == null)
        continue;

      Query query = trainingSet.get(iq);

      HashMap<Document, Double> ranks = new HashMap<>();
      HashMap<Document, Double> lambdas = new HashMap<>();
      HashMap<Document, Double> pws = new HashMap<>();
      HashMap<Document, Double> logs = new HashMap<>();
      double N = idcg(query.getDocList(), query.getDocList().size());


      List<Document> sorted = sortP(query);

      for (int i = 0; i < sorted.size(); i++) {
        Document doc = sorted.get(i);
        ranks.put(doc, ranker.forwardProp(doc));
        lambdas.put(doc, 0d);
        pws.put(doc, Math.pow(2, doc.getLabel()) - 1);
        logs.put(doc, 1 / Math.log(i + 2));
      }

      for (Document[] pair : trainingPairs.get(iq)) {
        double dNCG = N * (pws.get(pair[0]) - pws.get(pair[1])) * (logs.get(pair[0]) - logs.get(pair[1]));
        double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj)
        double lambda = Math.abs(new Activation.Sigmoid().output(diff) * dNCG);
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

  /**
   * This method returns the ideal DCG given a list of documents.
   * @param docList List of documents for which ideal DCG is desired.
   * @param position position = k in DCG@k
   * @return ideal DCG.
   */
  private double idcg(List<Document> docList, int position) {
    List<Document> docsRanks = new ArrayList<>(docList);
    docsRanks.sort(Comparator.comparingInt(Document::getLabel).reversed());
    double sum = 0;
    if (position > -1) {
      final int pos = Math.min(position, docsRanks.size());
      for (int i = 0; i < pos; i++) {
        sum += (Math.pow(2, docsRanks.get(i).getLabel()) - 1) / Math.log(i + 2);
      }
    }
    return sum * Math.log(2);  //Change of base
  }


}
