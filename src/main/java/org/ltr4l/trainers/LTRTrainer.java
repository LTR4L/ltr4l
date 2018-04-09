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

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.ltr4l.Ranker;
import org.ltr4l.evaluation.DCG;
import org.ltr4l.evaluation.RankEval;
import org.ltr4l.evaluation.RankEval.RankEvalFactory;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Report;

/**
 * Abstract class used for training the model held by Rankers.
 * This class is also the parameter holder.
 *
 * train() must be implemented based on algorithm used.
 */
public abstract class LTRTrainer<R extends Ranker, C extends Config> {
  protected final int epochNum;
  protected final List<Query> trainingSet;
  protected final List<Query> validationSet;
  protected double maxScore;
  protected final Report report;
  protected final R ranker;
  protected final C config;
  protected final Error errorFunc;
  protected final int batchSize;
  protected final int evalK;
  protected final String modelFile;
  protected final RankEval eval;

  LTRTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    this.config = getConfig(reader);
    config.overrideBy(override);     // TODO: want to use generic C instead of Config
    epochNum = config.numIterations;
    trainingSet = training.getQueries();
    validationSet = validation.getQueries();
    maxScore = 0d;
    ranker = constructRanker();
    assert(config.batchSize >= 0);
    batchSize = config.batchSize;
    eval = getEvaluator(config);
    evalK = getEvaluatorAtK(config);
    modelFile = getModelFile(config);
    this.report = Report.getReport(config);
    this.errorFunc = makeErrorFunc();
  }

  private static RankEval getEvaluator(Config config){
    if (config.evaluation == null || config.evaluation.evaluator == null || config.evaluation.evaluator.equals(""))
      return new DCG.NDCG();
    final String evaluator = config.evaluation.evaluator;
    return RankEvalFactory.get(evaluator);
  }

  private static int getEvaluatorAtK(Config config){
    final int K_DEFAULT = 10;
    if(config.evaluation == null || config.evaluation.params == null) return K_DEFAULT;
    return Config.getInt(config.evaluation.params, "k", K_DEFAULT);
  }

  private static String getModelFile(Config config){
    if(config.model == null || config.model.file == null || config.model.file.isEmpty())
      return Config.Model.DEFAULT_MODEL_FILE;
    return config.model.file;
  }

  private static String getReportFile(Config config){
    return (config.report == null) ? null : config.report.file;
  }

  public R getRanker() {
    return ranker;
  }

  abstract double calculateLoss(List<Query> queries);

  /**
   * This method is used to assign errorFunc.
   * Child classes must specify which error they will use.
   * @return Implementation of Error
   */
  protected abstract Error makeErrorFunc();

  public double[] calculateLoss() {
    return new double[]{calculateLoss(trainingSet), calculateLoss(validationSet)};
  }

  public void validate(int iter, int pos) {
    double newScore = eval.calculateAvgAllQueries(ranker, validationSet, pos);
    if (newScore > maxScore) {
      maxScore = newScore;
    }
    double[] losses = calculateLoss();
    report.log(iter, newScore, losses[0], losses[1]);
  }

  public abstract void train();

  public void trainAndValidate() {
    for (int i = 1; i <= epochNum; i++) {
      train();
      validate(i, evalK);
    }
    report.close();
    try {
      ranker.writeModel(config, modelFile);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  protected abstract <R extends Ranker> R constructRanker();

  public abstract <C extends Config> Class<C> getConfigClass();

  <C extends Config> C getConfig(Reader reader){
    ObjectMapper mapper = new ObjectMapper();
    try {
      return mapper.readValue(reader, getConfigClass());
    } catch (IOException e) {
      throw new IllegalArgumentException(e);
    }
  }

  public static class TrainerFactory {

    /**
     * This returns the appropriate implementation of Trainer depending on the algorithm.
     * @param algorithm Algorithm/implementation to be used.
     * @param trainingSet The QuerySet containing the data to be used for training.
     * @param validationSet The QuerySet containing the data to be used for validation.
     * @param configFile The Config file containing parameters needed for Ranker class.
     * @return new class which implements trainer.
     */
    public static LTRTrainer getTrainer(String algorithm, QuerySet trainingSet, QuerySet validationSet, String configFile, Config override) {
      try(Reader reader = new FileReader(configFile)){
        switch (algorithm.toLowerCase()) {
          case "prank":
            return new PRankTrainer(trainingSet, validationSet, reader, override);
          case "oap":
            return new OAPBPMTrainer(trainingSet, validationSet, reader, override);
          case "ranknet":
            return new RankNetTrainer(trainingSet, validationSet, reader, override);
          case "franknet":
            return new FRankTrainer(trainingSet, validationSet, reader, override);
          case "lambdarank":
            return new LambdaRankTrainer(trainingSet, validationSet, reader, override);
          case "nnrank":
            return new NNRankTrainer(trainingSet, validationSet, reader, override);
          case "sortnet":
            return new SortNetTrainer(trainingSet, validationSet, reader, override);
          case "listnet":
            return new ListNetTrainer(trainingSet, validationSet, reader, override);
          default:
            return null;
        }
      }
      catch (IOException e){
        throw new IllegalArgumentException(e);
      }
    }
  }
}
