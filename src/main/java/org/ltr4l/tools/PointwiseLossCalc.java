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
import org.ltr4l.nn.ListNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;

import java.util.List;
import java.util.Objects;

public abstract class PointwiseLossCalc<R extends Ranker> implements LossCalculator<R> {
  protected final List<Query> trainingSet;
  protected final List<Query> validationSet;

  protected PointwiseLossCalc(List<Query> trainingSet, List<Query> validationSet){
    this.trainingSet = trainingSet;
    this.validationSet = validationSet;
  }

  @Override
  public double calculateLoss(DataSet type, R ranker){
    Objects.requireNonNull(type);
    switch(type){
      case TRAINING:
        return calculateLoss(trainingSet, ranker);
      case VALIDATION:
        return calculateLoss(validationSet, ranker);
      default:
        throw new IllegalArgumentException();
    }
  }

  protected abstract double calculateLoss(List<Query> queries, R ranker);

  public static class StandardPointLossCalc <R extends Ranker> extends PointwiseLossCalc<R> {
    protected final Error errorFunc;

    public StandardPointLossCalc(List<Query> trainingSet, List<Query> validationSet, Error errorFunc){
      super(trainingSet, validationSet);
      this.errorFunc = errorFunc;
    }

    @Override
    protected double calculateLoss(List<Query> queries, R ranker) {
      double loss = 0d;
      for (Query query : queries) {
        List<Document> docList = query.getDocList();
        loss += docList.stream().mapToDouble(doc -> errorFunc.error(ranker.predict(doc.getFeatures()), doc.getLabel())).sum() / docList.size();
      }
      return loss / queries.size();
    }
  }

  public static class ListNetLossCalc extends StandardPointLossCalc<ListNetMLP>{

    public ListNetLossCalc(List<Query> trainingSet, List<Query> validationSet, Error errorFunc){
      super(trainingSet, validationSet, errorFunc);
    }

    @Override
    protected double calculateLoss(List<Query> queries, ListNetMLP ranker){
      double loss = 0;
      for (Query query : queries) {
        double targetSum = query.getDocList().stream().mapToDouble(i -> Math.exp(i.getLabel())).sum();
        double outputSum = query.getDocList().stream().mapToDouble(i -> Math.exp(ranker.forwardProp(i))).sum();
        double qLoss = query.getDocList().stream().mapToDouble(i -> errorFunc.error( //-Py(log(Pfx))
            Math.exp(ranker.forwardProp(i)) / outputSum, //output: exp(f(x)) / sum(f(x))
            i.getLabel() / targetSum))                 //target: y / sum(exp(y))
            .sum(); //sum over all documents                // Should it be exp(y)/sum(exp(y))?
        loss += qLoss;
      }
      return loss / queries.size();
    }

  }
}
