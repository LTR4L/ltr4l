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
import org.ltr4l.tools.Error;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class SMO extends Solver {
  protected final List<Double> lagrangeMults;

  public SMO(AbstractSVM.SVMConfig config, List<Document> trainingData) {
    super(config, trainingData);
    SVMInitializer init = new SVMInitializer(config.getSVMWeightInit());
    lagrangeMults = init.makeInitialWeights(trainingData.size()); //lagrange mult for every data point
  }

  @Override
  public void trainEpoch(Error error) {
  }

  @Override
  protected void iterate(List<Double> features, Error error, double output, double target){
  }

  @Override
  public List<Double> getWeights() {
    List<Double> weights = new ArrayList<>(Collections.nCopies(trainingData.get(0).getFeatureLength(), 0d));
    for (int i = 0; i < lagrangeMults.size(); i++) {
      double alpha = lagrangeMults.get(i);
      if(alpha == 0)
        continue;
      List<Double> grad = VectorMath.scalarMult(alpha, trainingData.get(i).getFeatures());
      weights = VectorMath.add(weights, grad);
    }
    return weights;
  }

  @Override
  public double getBias() {
    int i = 0;
    while(i < lagrangeMults.size() && lagrangeMults.get(i) != 0d)
      i++;
    if (i > lagrangeMults.size())
      throw new IllegalArgumentException("no valid support vector found...");
    Document supportVec = trainingData.get(i);
    return VectorMath.dot(getWeights(), supportVec.getFeatures()) - supportVec.getLabel();
/*    Document[] supportVecs = findSupportVectorPair();
    double s1Prod = kernel.similarityK(this.getWeights(), supportVecs[0].getFeatures(), kParams);
    double s2Prod = kernel.similarityK(this.getWeights(), supportVecs[1].getFeatures(), kParams);
    return - (s1Prod + s2Prod) / 2;*/
  }

  @Override
  public void updateWeights(double lrRate) {

  }

/*  protected Document[] findSupportVectorPair(){
    Document[] pair = new Document[2];
    for(int i = 0; i < lagrangeMults.size(); i++) {
      if (pair[0] != null && pair[1] != null)
        return pair;
      double alpha = lagrangeMults.get(i);
      if(alpha == 0)
        continue;
      Document doc = trainingData.get(i);
      if(doc.getLabel() == -1)
        pair[0] = doc;
      else
        pair[1] = doc;
    }
    throw new IllegalArgumentException("No valid support vector pair...");
  }*/

}
