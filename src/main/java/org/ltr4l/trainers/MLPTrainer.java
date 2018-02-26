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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

import org.ltr4l.nn.MLP;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import org.ltr4l.tools.Model;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

abstract class MLPTrainer extends LTRTrainer {
  protected MLP mlp;
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;

  MLPTrainer(QuerySet training, QuerySet validation, Config config) {
    this(training, validation, config, false);
  }

  //This constructor exists solely for the purpose of child classes
  //It gives child classes the ability to assign an extended MLP.
  MLPTrainer(QuerySet training, QuerySet validation, Config config, boolean hasOtherMLP) {
    super(training, validation, config.getNumIterations());
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
    if (!hasOtherMLP) {
      int featureLength = trainingSet.get(0).getFeatureLength();
      NetworkShape networkShape = config.getNetworkShape();
      Optimizer.OptimizerFactory optFact = config.getOptFact();
      Regularization regularization = config.getReguFunction();
      String weightModel = config.getWeightInit();
      mlp = new MLP(featureLength, networkShape, optFact, regularization, weightModel);
    }
  }

  protected double calculateLoss(List<Query> queries) {
    // Note: appears to be just as fast use of nested loops without streams.
    // However, I have not tested it thoroughly.
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> new Error.Square().error(mlp.predict(doc), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  @Override
  protected void logWeights(Model model){
    model.log(mlp.getBestWeights());
  }

  @Override
  protected void updateBestWeights(){
    mlp.recordWeights();
  }

  @Override
  public List<Document> sortP(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    ranks.sort(Comparator.comparingDouble(mlp::predict).reversed());
    return ranks;
  }
}
