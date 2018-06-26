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
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.query.RankedDocs;
import org.ltr4l.tools.*;
import org.ltr4l.tools.Error;

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class AdaBoostTrainer extends AbstractTrainer<AdaBoost, RankBoost.RankBoostConfig> {
  private final ABDistribution ABDistribution;
  private final List<RankedDocs> rTrainingSet; //Contains doc lists sorted by label. Queries with no pairs of differing labels should be removed.

  public AdaBoostTrainer(QuerySet training, QuerySet validation, Reader reader, Config override){
    super(training, validation, reader, override);
    rTrainingSet = new ArrayList<>();
    for(Query query : trainingSet) {
      //Sort into correct rank. This is not just for initialization, but also for later calculation.
      RankedDocs rDocs = new RankedDocs(query.getDocList());
      if(rDocs.getLabel(0) == rDocs.getLabel(rDocs.size() - 1)) continue;
      rTrainingSet.add(rDocs);
    }
    ABDistribution = new ABDistribution(rTrainingSet);
  }

  @Override
  protected Error makeErrorFunc() {
    return StandardError.ENTROPY;
  }

  @Override
  protected LossCalculator makeLossCalculator(){
    return new PointwiseLossCalc.StandardPointLossCalc<>(ranker, trainingSet, validationSet, StandardError.SQUARE);
  }

  @Override
  public void train() {
    WeakLearner wl = AdaWeakLearner.findWeakLearner(ABDistribution.getFullDist(), rTrainingSet, config.getNumSteps());
    ranker.addLearner(wl);
    ABDistribution.update(wl, rTrainingSet);
  }

  @Override
  protected AdaBoost constructRanker() {
    return new AdaBoost();
  }

  @Override
  public Class<RankBoost.RankBoostConfig> getConfigClass() {
    return getCC();
  }

  public static Class<RankBoost.RankBoostConfig> getCC(){
    return RankBoost.RankBoostConfig.class;
  }
}
