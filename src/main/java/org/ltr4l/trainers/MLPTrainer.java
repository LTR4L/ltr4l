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

import org.ltr4l.nn.MLP;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

/**
 * The basic implementation of LTRTrainer for classes which use Multi-Layer Perceptron rankers.
 * As the training method can be different depending on the algorithm used,
 * the method train() must be implemented by child classes.
 */
abstract class MLPTrainer<M extends MLP> extends LTRTrainer<M> {
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;
  //protected Config config;

  MLPTrainer(QuerySet training, QuerySet validation, Config config) {
    this(training, validation, config, false);
  }

  //This constructor exists solely for the purpose of child classes
  //It gives child classes the ability to assign an extended MLP.
  MLPTrainer(QuerySet training, QuerySet validation, Config config, boolean hasOtherMLP) {
    super(training, validation, config);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Square(); //Default square error
  }

  @Override
  protected MLP getRanker(){
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new MLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  protected double calculateLoss(List<Query> queries) {
    // Default square error.
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> errorFunc.error(ranker.predict(doc.getFeatures()), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

}
