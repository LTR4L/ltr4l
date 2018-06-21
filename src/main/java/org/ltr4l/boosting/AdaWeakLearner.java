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

import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.ArrayList;
import java.util.List;

public class AdaWeakLearner extends WeakLearner {

  public static WeakLearner findWeakLearner(double[][] distribution, List<RankedDocs> queries, int numSteps){ //For Adaboost.
    RankBoostTools tools = new RankBoostTools(distribution, queries);
    List<Document> docs = new ArrayList<>();
    queries.forEach(docs::addAll);
    OptimalLeafLoss optLoss = tools.findMinLeafThreshold(docs, numSteps);
    double r = 1 / optLoss.getMinLoss();
    double alpha = 0.5 * Math.log((1 + r) / (1 - r));
    return new AdaWeakLearner(optLoss.getOptimalFeature(), optLoss.getOptimalThreshold(), alpha);
  }

  protected AdaWeakLearner(int fid, double threshold, double alpha) {
    super(fid, threshold, alpha);
  }

  @Override
  protected int calculateScore(List<Double> features){
    return features.get(fid) < threshold ? -1 : 1;
  }
}
