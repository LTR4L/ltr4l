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

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.tools.Error;

import java.util.List;

public abstract class Solver {
  protected int numTrained;
  protected double bestMetric;
  protected int batchSize;
  protected double lrRate;
  protected final Kernel kernel;
  protected final KernelParams kParams;
  protected final List<Query> trainingQueries;

  protected Solver(AbstractSVM.SVMConfig config, List<Query> trainingQueries){
    kernel = config.getKernel();
    kParams = new KernelParams();
    batchSize = config.batchSize;
    lrRate = config.getLearningRate();
    this.trainingQueries = trainingQueries;
    numTrained = 0;
    bestMetric = 0d;
  }

  public abstract void trainEpoch(Error error);

  protected void iterate(Document doc, Error error) {
    List<Double> features = doc.getFeatures();
    double output = this.predict(features);
    double target = doc.getLabel();
    iterate(features, error, output, target);
  }

  protected abstract void iterate(List<Double> Features, Error error, double output, double target);
  public abstract List<Double> getWeights();
  public abstract double getBias();
  public abstract void updateWeights(double lrRate);

  public abstract double predict(List<Double> features);

  public static class Factory{
    public static Solver get(AbstractSVM.SVMConfig config, List<Query> trainingData) {
      Solver.Type type = config.getOptimizer();
      switch(type) {
        case sgd:
          return new SGD(config, trainingData);
        case smo:
          return new SMO(config, trainingData);
        default:
          return new SGD(config, trainingData);
      }
    }
  }

  public static enum Type {
    sgd, smo;

    public static Solver.Type get(String type){
      for(Solver.Type solver : Solver.Type.values())
        if(solver.name().equals(type))
          return solver;
      System.err.println("Invalid Solver (Optimizer) provided. Will use SGD.");
      return sgd;
    }
  }

}
