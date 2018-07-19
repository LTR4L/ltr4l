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

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.svm.*;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.LossCalculator;
import org.ltr4l.tools.StandardError;

import java.util.List;

public class RankSVMTrainer<C extends AbstractSVM.SVMConfig> extends AbstractTrainer<LinearSVM, C> {
  protected final List<Query> pwTraining;
  protected final List<Query> pwValidation;
  protected final SVMOptimizer optimizer;
  protected double prevBias; //Used for reversion, if direct metric evaluation (unrelated to loss) is desired...
  protected List<Double> prevWeights; //Used for reversion, if direct metric evaluation is desired...
  protected double lrRate;

  protected RankSVMTrainer(List<Query> training, List<Query> validation, C config, LinearSVM ranker, Error errorFunc, LossCalculator lossCalc) {
    super(training, validation, config, ranker, errorFunc, lossCalc);
    pwTraining = PairwiseQueryCreator.createQueries(training);
    pwValidation = PairwiseQueryCreator.createQueries(validation);
    optimizer = config.getOptimizer();
    lrRate = config.getLearningRate();
  }

  public RankSVMTrainer(List<Query> training, List<Query> validation, C config){
    this(training, validation, config, new LinearSVM(Kernel.Type.LINEAR, new SVMInitializer(config.getSVMWeightInit()), training.get(0).getFeatureLength()), StandardError.HINGE, null);
  }

  @Override
  public void train() {
    int numTrained = 0;
    for (int qid = 0; qid < pwTraining.size(); qid++){
      List<Document> query = pwTraining.get(qid).getDocList();
      for (int vecId = 0; vecId < query.size(); vecId++) {
        Document doc = query.get(vecId);
        double output = ranker.predict(doc.getFeatures());
        double target = doc.getLabel();
        double loss = errorFunc.error(output, target);
        if (loss <= 0)
          continue;
        ranker.optimize(optimizer, errorFunc, output, target);
        numTrained++;
        if (batchSize == 0 || numTrained % batchSize == 0) {
          //TODO: modify learning rate
          ranker.updateWeights(lrRate);
        }
      }
    }
    if (batchSize != 0 && numTrained % batchSize != 0)
      ranker.updateWeights(lrRate);
  }
}
