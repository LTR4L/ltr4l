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
package org.ltr4l.svm;

import org.ltr4l.evaluation.RankEval;
import org.ltr4l.query.Query;
import org.ltr4l.tools.Error;

import java.util.List;

public abstract class Solver<R extends AbstractSVM> {
  protected final int batchSize;
  protected int numTrained;
  protected R svmRanker;
  protected double bestMetric;
  protected final RankEval eval;

  protected Solver(R svmRanker, RankEval eval, int batchSize){
    this.svmRanker = svmRanker;
    this.eval = eval;
    this.batchSize = batchSize;
    numTrained = 0;
    bestMetric = 0d;
  }

  public abstract void optimize(AbstractSVM svm, Error errorFunc, List<Query> training, int batchSize);

  public static class Factory{
    public static Solver get(String string){
      return null; //TODO: implement...
    }
  }

  public static class SGD <L extends LinearSVM> extends Solver<L> {

    public SGD(L lSvmRanker, RankEval eval, int batchSize){
      super(lSvmRanker, eval, batchSize);
    }

    @Override
    public void optimize(AbstractSVM svm, Error errorFunc, List<Query> training, int batchSize){ }
  }


}
