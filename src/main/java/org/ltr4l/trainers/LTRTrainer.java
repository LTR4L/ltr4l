package org.ltr4l.trainers;

import java.util.List;

import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.RankEval;
import org.ltr4l.tools.Report;

public abstract class LTRTrainer implements Trainer {
  protected int epochNum;
  protected List<Query> trainingSet;
  protected List<Query> validationSet;
  double maxScore;
  protected final Report report;

  LTRTrainer(QuerySet training, QuerySet validation, int epochNum) {
    this.epochNum = epochNum;
    trainingSet = training.getQueries();
    validationSet = validation.getQueries();
    maxScore = 0d;
    this.report = Report.getReport();  // TODO: use default Report for now...
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
  }
}
