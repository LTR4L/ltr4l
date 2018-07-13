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

import java.io.IOException;
import java.io.Writer;
import java.util.List;

public class LinearSVM<C extends AbstractSVM.SVMConfig> extends AbstractSVM<C> {
  protected double bias;
  protected final List<Double> weights;
  protected double db;
  protected List<Double> dw;

  protected LinearSVM(Kernel kernel, SVMInitializer init, List<Double> weights, double bias){
    super(kernel);
    this.weights = weights;
    this.bias = bias;
  }

  @Override
  public void writeModel(C config, Writer writer) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public double predict(List<Double> features) {
    return VectorMath.dot(features, weights) + params.getC();
  }

}
