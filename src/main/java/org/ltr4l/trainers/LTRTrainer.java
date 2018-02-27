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

import org.ltr4l.nn.Ranker;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.RankEval;
import org.ltr4l.tools.Report;

public abstract class LTRTrainer implements Trainer {
  protected int epochNum;
  protected List<Query> trainingSet;
  protected List<Query> validationSet;
  double maxScore;
  protected final Report report;
  protected Ranker ranker;
  protected Config config;

  LTRTrainer(QuerySet training, QuerySet validation, Config config) {
    this.config = config;
    epochNum = config.getNumIterations();
    trainingSet = training.getQueries();
    validationSet = validation.getQueries();
    maxScore = 0d;
    this.report = Report.getReport();  // TODO: use default Report for now...
    ranker = null;
  }

  abstract double calculateLoss(List<Query> queries);

  @Override
  public double[] calculateLoss() {
    return new double[]{calculateLoss(trainingSet), calculateLoss(validationSet)};
  }

  @Override
  public void validate(int iter, int pos) {
    double newScore = RankEval.ndcgAvg(this, validationSet, pos);
    if (newScore > maxScore) {
      maxScore = newScore;
    }
    double[] losses = calculateLoss();
    report.log(iter, newScore, losses[0], losses[1]);
  }

  @Override
  public void trainAndValidate() {
    for (int i = 1; i <= epochNum; i++) {
      train();
      validate(i);
    }
    report.close();
    if (ranker == null) ranker = getRanker();
    ranker.writeModel(config.getProps());
  }

  abstract Ranker getRanker();

  protected abstract void logWeights();

}
