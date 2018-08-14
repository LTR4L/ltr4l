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

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class RankSVMTrainer extends AbstractTrainer<SVM, AbstractSVM.SVMConfig> {
  protected final List<Query> pwTraining;
  protected final List<Query> pwValidation;
  protected double lrRate;
  protected final boolean optMetric;

  protected RankSVMTrainer(List<Query> training, List<Query> validation, AbstractSVM.SVMConfig config, SVM ranker, Error errorFunc, LossCalculator lossCalc) {
    super(training, validation, config, ranker, errorFunc, lossCalc);
    pwTraining = PairwiseQueryCreator.createQueries(training); //Pairs with same labels / queries with only one label are thrown out...
    pwValidation = PairwiseQueryCreator.createQueries(validation);
    lrRate = config.getLearningRate();
    optMetric = config.getMetricOption();
  }

  public RankSVMTrainer(List<Query> training, List<Query> validation, AbstractSVM.SVMConfig config){
    this(
        training,
        validation,
        config,
        new SVM( config, training.stream().flatMap(q -> q.getDocList().stream()).collect(Collectors.toCollection(ArrayList::new))), //TODO: dimension will change depending on solver.....
        StandardError.HINGE,
        null);
  }

  @Override
  public void validate(int iter, int pos) { //TODO: IMPLEMENT CALCULATE LOSS
    double newScore = eval.calculateAvgAllQueries(ranker, pwValidation, pos);
    if (newScore > maxScore) {
      maxScore = newScore;
    }
    double[] losses = new double[]{0d, 0d};
    report.log(iter, newScore, losses[0], losses[1]);
  }

  @Override
  public void train() {
    ranker.optimize(errorFunc);
  }

/*  protected void updateRankerWeights(){
    if (optMetric)
      optimizeToMetric();
    else
      ranker.updateWeights(lrRate);
  }

  protected void optimizeToMetric(){ //This is conducted outside of ranker, as all documents are required to calculate metric
    List<Double> prevWeights = ranker.getWeights();
    double prevBias = ranker.getBias();
    optimizeToMetric(prevWeights, prevBias);
  }

  protected void optimizeToMetric(List<Double> prevWeights, double prevBias){
    ranker.updateWeights(lrRate);
    double newScore = eval.calculateAvgAllQueries(ranker, pwValidation, evalK);
    if (newScore > maxScore)
      maxScore = newScore;
    else
      ranker.revertWeights(prevWeights, prevBias);
  }*/

}
