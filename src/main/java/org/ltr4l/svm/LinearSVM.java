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

import org.ltr4l.tools.Error;

import java.io.IOException;
import java.io.Writer;
import java.util.List;

public class LinearSVM<C extends AbstractSVM.SVMConfig> extends AbstractSVM<C> {
  protected double bias;
  protected final List<Double> weights;
  protected double db;
  protected List<Double> dw;
  protected int numIter;

  public LinearSVM(Kernel kernel, SVMInitializer init, int dim){
    super(kernel);
    weights = init.makeInitialWeights(dim);
    bias = init.getBias();
    numIter = 0;
  }

  @Override
  public void writeModel(C config, Writer writer) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public double predict(List<Double> features) {
    return VectorMath.dot(features, weights) + params.getC();
  }

  @Override
  public void optimize(SVMOptimizer optimizer, Error error, double output, double target){
    throw new UnsupportedOperationException();
  }

  public void updateWeights(double lrRate){
    throw new UnsupportedOperationException();
  }

  public double getBias() {
    return bias;
  }

  public void setBias(double bias) {
    this.bias = bias;
  }

  public List<Double> getWeights() {
    return weights;
  }

  public double getDb() {
    return db;
  }

  public void setDb(double db) {
    this.db = db;
  }

  public List<Double> getDw() {
    return dw;
  }

  public void setDw(List<Double> dw) {
    this.dw = dw;
  }
}
