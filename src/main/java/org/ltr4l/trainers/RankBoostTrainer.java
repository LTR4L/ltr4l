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

import org.ltr4l.boosting.RBDistribution;
import org.ltr4l.boosting.RankBoost;
import org.ltr4l.boosting.WeakLearner;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.query.RankedDocs;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class RankBoostTrainer extends AbstractTrainer<RankBoost, RankBoost.RankBoostConfig>{
  private final RBDistribution distribution;
  private final List<RankedDocs> rTrainingSet; //Contains doc lists sorted by label. Queries with no pairs of differing labels should be removed.

  public RankBoostTrainer(QuerySet training, QuerySet validation, Reader reader, Config override){
    super(training, validation, reader, override);
    rTrainingSet = new ArrayList<>();
    for(Query query : trainingSet) {
      //Sort into correct rank. This is not just for initialization, but also for later calculation.
      RankedDocs rDocs = new RankedDocs(query.getDocList());
      if(rDocs.getLabel(0) == rDocs.getLabel(rDocs.size() - 1)) continue;
      rTrainingSet.add(rDocs);
    }
    distribution = RBDistribution.getInitDist(rTrainingSet);
  }

  @Override
  double calculateLoss(List<Query> queries) { //TODO: Make a LossCalculation class to be used by all trainers
    double loss = 0d;
    int pairs = 0;
    for(Query query : queries){
      List<Document> qDocs = query.getDocList();
      for(int i = 0; i < qDocs.size(); i++){ //TODO: Speed up
        for(int j = 0; j < qDocs.size(); j++){
          int lDiff = qDocs.get(i).getLabel() - qDocs.get(j).getLabel();
          if(lDiff <= 0 ) continue;
          double scoreDiff = ranker.predict(qDocs.get(i).getFeatures()) - ranker.predict(qDocs.get(j).getFeatures());
          if(lDiff * scoreDiff <= 0d) loss++;
          pairs++;
        }
      }
    }
    return loss / pairs;
  }

  @Override
  protected Error makeErrorFunc() {
   return new Error.Entropy();
  }

  @Override
  public void train() {
    //One iteration of training.
    WeakLearner wl = WeakLearner.findWeakLearner(distribution, rTrainingSet, config.getNumSteps());
    ranker.addLearner(wl);
    distribution.update(wl, rTrainingSet);
  }

  @Override
  protected RankBoost constructRanker() {
    return new RankBoost();
  }

  @Override
  public Class<RankBoost.RankBoostConfig> getConfigClass() {
    return getCC();
  }

  public static Class<RankBoost.RankBoostConfig> getCC(){
    return RankBoost.RankBoostConfig.class;
  }

}
