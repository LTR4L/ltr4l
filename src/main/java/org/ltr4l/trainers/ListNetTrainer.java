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

import org.ltr4l.Ranker;
import org.ltr4l.nn.*;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

/**
 * ListNetTrainer is an extension of LTRTrainer.
 * Despite note extending MLPTrainer, this trainer
 * trains an MLP network.
 */
public class ListNetTrainer extends LTRTrainer<ListNetMLP> {
  private double lrRate;
  private double rgRate;

  ListNetTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
  }

  @Override
  protected ListNetMLP constructRanker() {
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
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
  public void train() {
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        ranker.forwardProp(doc);
        ranker.backProp(doc.getLabel());
      }
      ranker.updateWeights(lrRate, rgRate);
    }
  }

  @Override
  protected double calculateLoss(List<Query> querySet) {
    double loss = 0;
    for (Query query : querySet) {
      double targetSum = query.getDocList().stream().mapToDouble(i -> Math.exp(i.getLabel())).sum();
      double outputSum = query.getDocList().stream().mapToDouble(i -> Math.exp(ranker.forwardProp(i))).sum();
      double qLoss = query.getDocList().stream().mapToDouble(i -> errorFunc.error( //-Py(log(Pfx))
          Math.exp(ranker.forwardProp(i)) / outputSum, //output: exp(f(x)) / sum(f(x))
          i.getLabel() / targetSum))                 //target: y / sum(exp(y))
          .sum(); //sum over all documents                // Should it be exp(y)/sum(exp(y))?
      loss += qLoss;
    }
    return loss / querySet.size();
  }



}

