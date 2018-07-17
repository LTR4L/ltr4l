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
import org.ltr4l.query.Query;
import org.ltr4l.query.RankedDocs;
import org.ltr4l.tools.*;

import java.util.ArrayList;
import java.util.List;

public class RankBoostTrainer extends AbstractTrainer<RankBoost, RankBoost.RankBoostConfig>{
  private final RBDistribution distribution;
  private final List<RankedDocs> rTrainingSet; //Contains doc lists sorted by label. Queries with no pairs of differing labels should be removed.

  public RankBoostTrainer(List<Query> training, List<Query> validation, RankBoost.RankBoostConfig config, RankBoost ranker){
    super(training, validation, config, ranker, StandardError.ENTROPY, new PairwiseLossCalc.RankBoostLossCalc(training, validation));
    rTrainingSet = new ArrayList<>();
    for(Query query : trainingSet) {
      //Sort into correct rank. This is not just for initialization, but also for later calculation.
      RankedDocs rDocs = new RankedDocs(query.getDocList());
      if(rDocs.getLabel(0) == rDocs.getLabel(rDocs.size() - 1)) continue;
      rTrainingSet.add(rDocs);
    }
    distribution = new RBDistribution(rTrainingSet);
  }

  public RankBoostTrainer(List<Query> training, List<Query> validation, RankBoost.RankBoostConfig config){
    this(training, validation, config, new RankBoost());
  }

  @Override
  public void train() {
    //One iteration of training.
    WeakLearner wl = WeakLearner.findWeakLearner(distribution, rTrainingSet, config.getNumSteps());
    ranker.addLearner(wl);
    distribution.update(wl, rTrainingSet);
  }
}
