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
package org.ltr4l.tools;

import org.ltr4l.Ranker;
import org.ltr4l.boosting.RankBoost;
import org.ltr4l.nn.RankNetMLP;
import org.ltr4l.nn.SortNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

public abstract class PairwiseLossCalc<R extends Ranker> implements LossCalculator{
  protected final R ranker;
  protected final List<Document[][]> trainingPairs;
  protected final List<Document[][]> validationPairs;

  protected PairwiseLossCalc(R ranker, List<Query> trainingSet, List<Query> validationSet){
    this.ranker = ranker;
    trainingPairs = trainingSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
    validationPairs = validationSet.stream().map(query -> query.orderDocPairs()).collect(Collectors.toList());
  }

  public List<Document[][]> getValidationPairs() {
    return validationPairs;
  }

  public List<Document[][]> getTrainingPairs() {
    return trainingPairs;
  }

  @Override
  public double calculateLoss(DataSet type){
    Objects.requireNonNull(type);
    switch (type){
      case TRAINING:
        return calculateLoss(trainingPairs);
      case VALIDATION:
        return calculateLoss(validationPairs);
      default:
        throw new IllegalArgumentException();
    }
  }

  protected abstract double calculateLoss(List<Document[][]> docPairs);

  public static class RankNetLossCalc<R extends Ranker> extends PairwiseLossCalc<R> {
    private final Error errorFunc;

    public RankNetLossCalc(R ranker, List<Query> trainingSet, List<Query> validationSet, Error errorFunc){
      super(ranker, trainingSet, validationSet);
      this.errorFunc = errorFunc;

    }

    @Override
    protected double calculateLoss(List<Document[][]> docPairs) {
      double loss = 0d;
      int processedQueryNum = 0;
      for (Document[][] query : docPairs) {
        if (query == null)
          continue;
        processedQueryNum++;
        double queryLoss = 0d;
        for (Document[] pair : query) {
          double s1 = ranker.predict(pair[0].getFeatures());
          double s2 = ranker.predict(pair[1].getFeatures());
          double output = Math.pow(1 + Math.exp(s2 - s1), -1); //double output = new Activation.Sigmoid().output(s1 - s2);
          queryLoss += errorFunc.error(output, 1d);
        }
        loss += queryLoss / query.length;
      }
      return loss / processedQueryNum;
    }
  }

  public static class SortNetLossCalc extends PairwiseLossCalc<SortNetMLP>{
    protected final Error errorFunc;
    protected final double[][] targets;

    public SortNetLossCalc(SortNetMLP ranker, List<Query> trainingSet, List<Query> validationSet, Error errorFunc, double[][] targets){
      super(ranker, trainingSet, validationSet);
      this.errorFunc = errorFunc;
      this.targets = targets;
    }

    @Override
    protected double calculateLoss(List<Document[][]> docPairs) {
      double loss = 0d;
      for (Document[][] pairs : docPairs) {
        if (pairs == null)
          continue;
        double queryLoss = 0d;
        for (Document[] pair : pairs) {
          List<Double> combinedFeatures = new ArrayList<>(pair[0].getFeatures());
          combinedFeatures.addAll(pair[1].getFeatures());
          ranker.forwardProp(combinedFeatures);
          double[] outputs = ranker.getOutputs();
          queryLoss += errorFunc.error(outputs[0], targets[0][0]);
          queryLoss += errorFunc.error(outputs[1], targets[0][1]);
        }
        loss += queryLoss / (double) pairs.length;
      }
      return loss / (double) docPairs.size();
    }
  }

  public static class RankBoostLossCalc<R extends Ranker> extends PairwiseLossCalc<R>{

    public RankBoostLossCalc(R ranker, List<Query> trainingSet, List<Query> validationSet){
      super(ranker, trainingSet, validationSet);
    }

    @Override
    protected double calculateLoss(List<Document[][]> docPairs) {
      double loss = 0d;
      int pairs = 0;
      for(Document[][] query : docPairs){
        if(query == null) continue; //Don't count unhelpful queries.
        for(Document[] pair : query){
          Document doc1 = pair[0]; //Note that docPairs are arranged so higher label is [0].
          Document doc2 = pair[1];
          double scoreDiff = ranker.predict(doc1.getFeatures()) - ranker.predict(doc2.getFeatures());
          if(scoreDiff <= 0) loss++;
          pairs++;
        }
      }
      return loss / pairs;
    }
  }

}
