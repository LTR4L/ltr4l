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

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class SGD extends Solver {
  protected final List<Double> weights;
  protected final List<Document> trainingData;
  protected List<Double> dw;
  protected double bias;
  protected double db;

  public SGD(AbstractSVM.SVMConfig config, List<Query> trainingQueries) {
    super(config, trainingQueries);
    SVMInitializer init = new SVMInitializer(config.getSVMWeightInit());
    trainingData = trainingQueries.stream().flatMap(q -> q.getDocList().stream()).collect(Collectors.toCollection(ArrayList::new));
    weights = init.makeInitialWeights(trainingQueries.get(0).getFeatureLength()); //linear SGD primal weights...
    bias = init.getBias();
    db = 0d;
    dw = new ArrayList<>(Collections.nCopies(weights.size(), 0d));
  }

  @Override
  public double predict(List<Double> features) {
    return kernel.similarityK(features, this.getWeights(), kParams.setC(getBias()));
  }

  @Override
  public void trainEpoch(Error error) {
    List<Document> data = new ArrayList<>(trainingData);
    Collections.shuffle(data);
    for (Document doc : data)
      iterate(doc, error);
    updateWeights(lrRate);
  }

  @Override
  protected void iterate(List<Double> features, Error error, double output, double target){
    if (error.error(output, target) <= 0)
      return;
    db += error.der(output, target);
    if (error.der(output, target) == 0d)
      throw new IllegalArgumentException();
    List<Double> dwNew = VectorMath.scalarMult(error.der(output, target), features);
    dw = VectorMath.add(dw, dwNew);
    numTrained++;
    if (batchSize != 0 && numTrained % batchSize == 0)
      updateWeights(lrRate); //TODO: modify learning rate
  }

  @Override
  public List<Double> getWeights() {
    return weights;
  }

  @Override
  public double getBias() {
    return bias;
  }

  @Override
  public void updateWeights(double lrRate) {
    assert(numTrained >= 0);
    if(numTrained == 0)
      return;
    bias -= lrRate * db / numTrained;
    for (int i = 0; i < weights.size(); i++) {
      double w = weights.get(i) - lrRate * dw.get(i) / numTrained;
      weights.set(i, w);
    }
    numTrained = 0;
  }

}
