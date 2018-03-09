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
import java.util.List;

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.RankEval;
import org.ltr4l.tools.Report;
import org.ltr4l.tools.Error;

/**
 * Abstract class used for training the model held by Rankers.
 * This class is also the parameter holder.
 *
 * train() must be implemented based on algorithm used.
 */
public abstract class LTRTrainer<R extends Ranker> implements Trainer {
  protected final int epochNum;
  protected final List<Query> trainingSet;
  protected final List<Query> validationSet;
  protected double maxScore;
  protected final Report report;
  protected R ranker;
  protected final Config config;
  protected final Error errorFunc;

  LTRTrainer(QuerySet training, QuerySet validation, Config config) {
    this.config = config;
    epochNum = config.getNumIterations();
    trainingSet = training.getQueries();
    validationSet = validation.getQueries();
    maxScore = 0d;
    ranker = constructRanker();
    this.report = Report.getReport();  // TODO: use default Report for now...
    this.errorFunc = makeErrorFunc();
  }

  abstract double calculateLoss(List<Query> queries);

  /**
   * This method is used to assign errorFunc.
   * Child classes must specify which error they will use.
   * @return Implementation of Error
   */
  protected abstract Error makeErrorFunc();

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
    ranker.writeModel(config.getProps());
  }

  abstract protected <R extends Ranker> R constructRanker();

  /**
   * Sorts the associated documents in a  query according to the ranker's model via predict method, from highest score to lowest.
   * For example, if a query has the following associated document list:
   * {(2)docA, (3)docB, (1)docC}
   * where the numbers in the parentheses are predicted scores,
   * sortP will return the following new list:
   * {docB, docA, docC}
   *
   * A new list is made in order to preserve the order of the original document list.
   *
   * sortP is currently also used to calculate NDCG, and thus a new sorted list should be used to avoid calculation errors.
   *
   * @param query
   * @return new sorted document list.
   */
  @Override
  public List<Document> sortP(Query query){
    List<Document> ranks = new ArrayList<>(query.getDocList());
    ranks.sort((docA, docB) -> Double.compare(ranker.predict(docB.getFeatures()), ranker.predict(docA.getFeatures()))); //reversed for high to low.
    return ranks;
  }

}
