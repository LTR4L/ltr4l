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

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.*;

public class OAPBPMTrainer extends LTRTrainer {
  final private OAPBPMRank ranker;
  private double maxScore;
  private final  List<Document> trainingDocList;

  OAPBPMTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config.getNumIterations());
    maxScore = 0d;
    ranker = new OAPBPMRank(trainingSet.get(0).getFeatureLength(), QuerySet.findMaxLabel(trainingSet), config.getPNum(), config.getBernNum());
    trainingDocList = new ArrayList<>();
    for (Query query : trainingSet)
      trainingDocList.addAll(query.getDocList());
  }

  @Override
  public void train() {
    for (Document doc : trainingDocList)
      ranker.updateWeights(doc);
/*    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        ranker.updateWeights(doc);
      }
    }*/
  }

  protected double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> new Error.SQUARE().error(ranker.predict(doc), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  @Override
  public List<Document> sortP(Query query) {
    List<Document> ranks = query.getDocList();
    ranks.sort(Comparator.comparingInt(ranker::predict).reversed());
    return ranks;
  }
}

class OAPBPMRank extends PRank {
  private List<PRank> pRanks;
  private final double bernProb;

  OAPBPMRank(int featureLength, int maxLabel, int pNumber, double bernNumber) {
    super(featureLength, maxLabel);
    pRanks = new ArrayList<>();
    for (int i = 0; i < pNumber; i++)
      pRanks.add(new PRank(featureLength, maxLabel));
    bernProb = bernNumber; //Note: must be between 0 and 1.
  }

  @Override
  public void updateWeights(Document doc) {
    for (PRank prank : pRanks) {
      //Will or will not present document to the perceptron.
      if (bernoulli() == 1) {
        int prediction = prank.predict(doc);
        int label = doc.getLabel();
        if (label != prediction) { //if the prediction is wrong, update that perceptron's weights
          prank.updateWeights(doc);
          for (int i = 0; i < weights.length; i++)
            weights[i] += prank.getWeights()[i] / (double) pRanks.size(); //and update overall weights
          for (int i = 0; i < thresholds.length; i++)
            thresholds[i] += prank.getThresholds()[i] / (double) pRanks.size();
        }
      }

    }
  }

  private int bernoulli() {
    return new Random().nextDouble() < bernProb ? 1 : 0;
  }

}
