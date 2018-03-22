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
import org.ltr4l.nn.ListNetMLP;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

/**
 * ListNetTrainer is an extension of LTRTrainer.
 * Despite note extending MLPTrainer, this trainer
 * trains an MLP network.
 */
public class ListNetTrainer extends MLPTrainer<ListNetMLP> {
  private double lrRate;
  private double rgRate;

  ListNetTrainer(QuerySet training, QuerySet validation, String file) {
    super(training, validation, file);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
  }

  @Override
  protected ListNetMLP constructRanker() {
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    networkShape.add(1, new Activation.Identity());
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new ListNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Entropy();
  }

  @Override
  public Class<MLPTrainer.MLPConfig> getConfigClass(){
    return MLPTrainer.MLPConfig.class;
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

