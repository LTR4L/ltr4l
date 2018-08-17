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
import org.ltr4l.tools.StandardError;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.Collectors;

public class SGD extends Solver {
  protected final List<Double> weights;
  protected final List<Document> trainingData;
  protected final Error errorFunc;
  protected List<Double> dw;
  protected double bias;
  protected double db;

  public SGD(AbstractSVM.SVMConfig config, List<Query> trainingQueries) {
    super(config, trainingQueries);
    SVMInitializer init = new SVMInitializer(config.getSVMWeightInit());
    trainingData = trainingQueries.stream().flatMap(q -> q.getDocList().stream()).collect(Collectors.toCollection(ArrayList::new));
    weights = init.makeInitialWeights(trainingQueries.get(0).getFeatureLength()); //linear SGD primal weights...
    errorFunc = StandardError.HINGE;
    bias = init.getBias();
    db = 0d;
    dw = new ArrayList<>(Collections.nCopies(weights.size(), 0d));
  }

  @Override
  public double predict(List<Double> features) {
    return kernel.similarityK(features, weights, kParams.setC(bias));
  }

  @Override
  public void trainEpoch() {
    List<Document> data = new ArrayList<>(trainingData);
    Collections.shuffle(data);
    for (Document doc : data)
      iterate(doc);
    updateWeights(lrRate);
  }

  protected void iterate(Document doc) {
    List<Double> features = doc.getFeatures();
    double output = this.predict(features);
    double target = doc.getLabel();
    iterate(features, output, target);
  }

  protected void iterate(List<Double> features, double output, double target){
    if (errorFunc.error(output, target) <= 0)
      return;
    db += errorFunc.der(output, target);
    if (errorFunc.der(output, target) == 0d)
      throw new IllegalArgumentException();
    List<Double> dwNew = VectorMath.scalarMult(errorFunc.der(output, target), features);
    dw = VectorMath.add(dw, dwNew);
    numTrained++;
    if (batchSize != 0 && numTrained % batchSize == 0)
      updateWeights(lrRate); //TODO: modify learning rate
  }

  private void updateWeights(double lrRate) {
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
