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

import java.io.IOException;
import java.io.Writer;
import java.util.List;

public class SVM<C extends AbstractSVM.SVMConfig> extends AbstractSVM<C> {
  protected final Solver solver;

  public SVM(SVMConfig config, List<Query> trainingData) {
    super(config.getKernel());
    solver = Solver.Factory.get(config, trainingData);
  }

  @Override
  public void writeModel(C config, Writer writer) throws IOException {
    throw new UnsupportedOperationException();
  }

  @Override
  public double predict(List<Double> features) {
    return solver.predict(features);
  }

  @Override
  public void optimize(){
    solver.trainEpoch();
  }

}
