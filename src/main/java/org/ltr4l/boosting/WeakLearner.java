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
package org.ltr4l.boosting;

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

public class WeakLearner extends Ranker<RankBoost.RankBoostConfig> {
  private final int fid;
  private final double threshold;
  private double alpha;

  public static WeakLearner findWeakLearner(RBDistribution dist, List<RankedDocs> queries){ // Here we want to find alpha and criteria for new weak learner
    //Note: The implementation here uses an approximation; see the third method of 3.2 in the original paper.
    double[][] potential = dist.calcPotential();
    RankBoostTools tools = new RankBoostTools(potential, queries);
    List<Document> docs = new ArrayList<>();
    queries.forEach(docs::addAll);
    OptimalLeafLoss optloss = tools.findMinLeafThreshold(docs, 10);
    double r = 1 / optloss.getMinLoss();
    double alpha = 0.5 * Math.log((1 + r) / (1 - r));
    return new WeakLearner(optloss.getOptimalFeature(), optloss.getOptimalThreshold(), alpha);
  }


  protected WeakLearner(int fid, double threshold, double alpha){
    this.fid = fid;
    this.threshold = threshold;
    this.alpha = alpha;
  }

  public int calculateScore(List<Double> features){
    return features.get(fid) < threshold ? 0 : 1;
  }

  public int getFid() {
    return fid;
  }

  @Override
  public void writeModel(RankBoost.RankBoostConfig config, Writer writer) throws IOException {

  }

  @Override
  public double predict(List<Double> features) {
    return calculateScore(features);
  }

  public double getAlpha() {
    return alpha;
  }
}
