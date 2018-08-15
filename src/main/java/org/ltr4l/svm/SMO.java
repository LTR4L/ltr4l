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

public class SMO extends Solver {
  protected final List<Double> lagrangeMults;
  protected final List<Document> trainingData;
  protected final double[] errorCache;
  protected final double lmConstraint;
  protected double bias;

  public SMO(AbstractSVM.SVMConfig config, List<Query> trainingQueries) {
    super(config, trainingQueries);
    SVMInitializer init = new SVMInitializer(config.getSVMWeightInit());
    trainingData = trainingQueries.stream().flatMap(q -> q.getDocList().stream()).collect(Collectors.toCollection(ArrayList::new));
    lagrangeMults = init.makeInitialWeights(trainingData.size()); //lagrange mult for every data point
    errorCache = new double[lagrangeMults.size()];
    lmConstraint = 0d;
  }

  @Override
  public double predict(List<Double> features) {
    double output = 0d;
    for (int i = 0; i < lagrangeMults.size(); i++) {
      Document doc = trainingData.get(i);
      double alpha = lagrangeMults.get(i); //Note: non-support vectors have zero lagrange multiplier
      output += doc.getLabel() * alpha * kernel.similarityK(doc.getFeatures(), features);
    }
    return output - getBias();
  }

  @Override
  public void trainEpoch(Error error) {

  }

  @Override
  protected void iterate(List<Double> features, Error error, double output, double target){
  }

  @Override
  public List<Double> getWeights() {
    List<Double> weights = new ArrayList<>(Collections.nCopies(trainingQueries.get(0).getFeatureLength(), 0d));
    for (int i = 0; i < lagrangeMults.size(); i++) {
      double alpha = lagrangeMults.get(i);
      if(alpha == 0)
        continue;
      List<Double> grad = VectorMath.scalarMult(alpha, trainingData.get(i).getFeatures());
      weights = VectorMath.add(weights, grad);
    }
    return weights;
  }

  private double calculateBias() {
    int i = 0;
    while (i < lagrangeMults.size() && lagrangeMults.get(i) != 0d)
      i++;
    if (i > lagrangeMults.size())
      throw new IllegalArgumentException("no valid support vector found...");
    Document supportVec = trainingData.get(i);
    bias = VectorMath.dot(getWeights(), supportVec.getFeatures()) - supportVec.getLabel();
    return bias;
  }

  @Override
  public double getBias() {
    return bias;
/*    Document[] supportVecs = findSupportVectorPair();
    double s1Prod = kernel.similarityK(this.getWeights(), supportVecs[0].getFeatures(), kParams);
    double s2Prod = kernel.similarityK(this.getWeights(), supportVecs[1].getFeatures(), kParams);
    return - (s1Prod + s2Prod) / 2;*/
  }

  @Override
  public void updateWeights(double lrRate) {

  }


  /**
   * Used to optimize lagrange multipliers i1 and i2; constraints on i2 are used to optimize.
   * Code from paper was used as reference.
   * @param i1
   * @param i2
   * @return
   */
  private boolean takeStep(int i1, int i2, double eps) {
    if (i1 == i2)
      return false;
    double alph1 = lagrangeMults.get(i1);
    double alph2 = lagrangeMults.get(i2);
    int y1 = trainingData.get(i1).getLabel();
    int y2 = trainingData.get(i2).getLabel();
    List<Double> feat1 = trainingData.get(i1).getFeatures();
    List<Double> feat2 = trainingData.get(i2).getFeatures();
    double E1 = predict(feat1) - y1; //TODO: Have error cache...
    double E2 = predict(feat1) - y2;
    int s = y1 * y2;
    double L = s == -1 ? Math.max(0d, alph2 - alph1) : Math.max(0, alph2 + alph1 - lmConstraint);
    double H = s == -1 ? Math.min(lmConstraint, lmConstraint + alph2 - alph1) : Math.min(lmConstraint, alph2 + alph1);
    if (L == H)
      return false;
    double k11 = kernel.similarityK(feat1, feat1);
    double k12 = kernel.similarityK(feat1, feat2);
    double k22 = kernel.similarityK(feat2, feat2);
    double eta = k11 + k22 - 2*k12;
    double a2; //to update alpha2
    if (eta > 0) { //in general, as long as the kernel obeys Mercer's conditions, eta should be positive!
      a2 = alph2 + y2 * (E1 - E2) / eta;
      if (a2 < L) a2 = L;
      else if (a2 > H) a2 = H;
    } else {
      double Lobj = obj(alph1, alph2, y1, y2, E1, E2, k11, k12, k22, L);
      double Hobj = obj(alph1, alph2, y1, y2, E1, E2, k11, k12, k22, H);
      if (Lobj < Hobj - eps)
        a2 = L;
      else if (Lobj > Hobj + eps)
        a2 = H;
      else
        a2 = alph2;
    }
    if (Math.abs(a2 - alph2) < eps * (a2 + alph2 + eps))
      return false;
    double a1 = alph1 + s * (alph2 - a2);
    //Compute new threshold...
    double b1 = E1 + y1 * (a1 - alph1) * k11 + y2 * (a2 - alph2) * k12 + bias;
    double b2 = E2 + y1 * (a1 - alph1) * k12 + y2 * (a2 - alph2) * k22 + bias;
    bias = (b1 + b2) / 2;
    //TODO: update error cache
    lagrangeMults.set(i1, a1);
    lagrangeMults.set(i2, a2);
    return true;
  }

  private int examineExample(int i2) {

  }

  private void updateErrorCache() {
    for (int i = 0; i < lagrangeMults.size(); i++) {
      errorCache[i] = calculateError(i);
    }
  }

  private double calculateError(int i2) {
    Document doc = trainingData.get(i2);
    return predict(doc.getFeatures()) - doc.getLabel();
  }

  private double obj(double alpha1, double alpha2, int y1, int y2, double E1, double E2, double k11, double k12, double k22, double bound) {
    int s = y1 * y2;
    double f1 = y1 * (E1 + bias) - alpha1 * k11 - s * alpha2 * k12;
    double f2 = y2 * (E2 + bias) - s * alpha1 * k12 - alpha2 * k22;
    double bound1 = alpha1 + s * (alpha2 - bound);
    return (bound1 * f1) + (bound * f2) + (bound1 * bound1 * k11)/2 + (bound * bound * k22)/2 + (s * bound * bound1 * k12);
  }

  private boolean kktAreSatisifiedBy(int i) {
    double alpha = lagrangeMults.get(i);
    double output = predict(trainingData.get(i).getFeatures());
    int label = trainingData.get(i).getLabel();
    if (alpha == 0 && label * output >= 1)
      return true;
    if ((alpha > 0 && alpha < lmConstraint) && label * output == 1)
      return true;
    return alpha == lmConstraint && label * output <= 1;
  }

  private int[] chooseLagrangePair() {

  }

/*  protected Document[] findSupportVectorPair(){
    Document[] pair = new Document[2];
    for(int i = 0; i < lagrangeMults.size(); i++) {
      if (pair[0] != null && pair[1] != null)
        return pair;
      double alpha = lagrangeMults.get(i);
      if(alpha == 0)
        continue;
      Document doc = trainingQueries.get(i);
      if(doc.getLabel() == -1)
        pair[0] = doc;
      else
        pair[1] = doc;
    }
    throw new IllegalArgumentException("No valid support vector pair...");
  }*/

}
