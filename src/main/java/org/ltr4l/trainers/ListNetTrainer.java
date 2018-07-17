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

import java.util.List;

import org.ltr4l.nn.ListNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.tools.*;
import org.ltr4l.tools.Error;

/**
 * ListNetTrainer is an extension of AbstractTrainer.
 * Despite note extending MLPTrainer, this trainer
 * trains an MLP network.
 *
 * Z. Cao, T. Qin, T. Liu, M. Tsai, and H. Li: Learning to rank: from pairwise approach to listwise
 * approach . Proceedings of the International Conference on Machine Learning. pp. 129–136, 2007.
 */
public class ListNetTrainer extends MLPTrainer<ListNetMLP> {
  private double lrRate;
  private double rgRate;

  ListNetTrainer(List<Query> training, List<Query> validation, MLPConfig config, ListNetMLP ranker, Error errorFunc, LossCalculator<ListNetMLP> lossCalc) {
    super(training, validation, config, ranker, errorFunc, lossCalc);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
  }

  ListNetTrainer(List<Query> training, List<Query> validation, MLPConfig config, ListNetMLP ranker){
    this(training,validation, config, ranker, StandardError.ENTROPY,
        new PointwiseLossCalc.ListNetLossCalc(training, validation, StandardError.ENTROPY));
  }

  ListNetTrainer(List<Query> training, List<Query> validation, MLPConfig config){
    this(training, validation, config, new ListNetMLP(training.get(0).getFeatureLength(), config));
  }

  @Override
  public void train() {
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        ranker.forwardProp(doc);
        ranker.backProp(doc.getLabel());
      }
      ranker.updateWeights(lrRate, rgRate);
    }
  }

}

