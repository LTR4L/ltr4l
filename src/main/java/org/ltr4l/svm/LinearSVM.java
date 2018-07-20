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
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class LinearSVM<C extends AbstractSVM.SVMConfig> extends AbstractSVM<C> {
  protected double bias;
  protected final List<Double> weights;
  protected double db;
  protected List<Double> dw;
  protected int accDer;

  public LinearSVM(Kernel kernel, SVMInitializer init, int dim){
    super(kernel);
    weights = init.makeInitialWeights(dim);
    bias = init.getBias();
    accDer = 0;
    db = 0;
    dw = new ArrayList<>(Collections.nCopies(weights.size(), 0d));
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
  public void optimize(List<Double> features, SVMOptimizer optimizer, Error error, double output, double target){
    //TODO: SGD hard coded...
    db += error.der(output, target);
    if (error.der(output, target) == 0d) throw new IllegalArgumentException();
    List<Double> dwNew = VectorMath.scalarMult(error.der(output, target), features);
    dw = VectorMath.add(dw, dwNew);
    accDer++;
  }

  public void updateWeights(double lrRate){
    assert (accDer  != 0);
    bias -= lrRate * db / accDer;
    for (int i = 0; i < weights.size(); i++) {
      double w = weights.get(i) - lrRate * dw.get(i) / accDer;
      weights.set(i, w);
    }
    accDer = 0;;
  }

  public double getBias() {
    return bias;
  }

  public List<Double> getWeights() {
    return new ArrayList<>(weights);
  }

  public void revertWeights(List<Double> prevWeights, double prevBias){
    assert(weights.size() == prevWeights.size());
    this.bias = prevBias;
    for(int i = 0; i < weights.size(); i++)
      this.weights.set(i, prevWeights.get(i));
  }

  public List<Double> getDw() {
    return dw;
  }

  public void setDw(List<Double> dw) {
    this.dw = dw;
  }
}
