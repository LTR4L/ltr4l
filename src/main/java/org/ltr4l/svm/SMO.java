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

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class SMO extends Solver {
  protected final List<Double> lagrangeMults;
  protected final List<Document> trainingData;
  protected double[] errorCache;
  protected final double lmConstraint;
  protected final double tolerance;
  protected final double eps;

  protected double bias;

  public SMO(AbstractSVM.SVMConfig config, List<Query> trainingQueries) {
    super(config, trainingQueries);
    SVMInitializer init = new SVMInitializer(config.getSVMWeightInit());
    trainingData = trainingQueries.stream().flatMap(q -> q.getDocList().stream()).collect(Collectors.toCollection(ArrayList::new));
    lagrangeMults = init.makeInitialWeights(trainingData.size()); //lagrange mult for every data point
    errorCache = initializeError();
    tolerance = 0.001;
    eps = 1e-8;
    lmConstraint = 0d;
  }

  private double[] initializeError() {
    double[] initialError = new double[trainingData.size()];
    System.out.printf("Initializing error cache for %d samples\n", initialError.length);
    for (int i = 0; i < initialError.length; i++)
      initialError[i] = calculateError(i);
    System.out.println("Error cache initialized...");
    return initialError;
  }

  private double calculateError(int i) {
    if(i % (trainingData.size() % 100 == 0 ? trainingData.size() / 100 : trainingData.size() / 100 + 1) == 0)
      System.out.print("#");
    Document doc = trainingData.get(i);
    return predict(doc.getFeatures()) - doc.getLabel();
  }

  @Override
  public double predict(List<Double> features) {
    double output = 0d;
    for (int i = 0; i < lagrangeMults.size(); i++) {
      Document doc = trainingData.get(i);
      double alpha = lagrangeMults.get(i); //Note: non-support vectors have zero lagrange multiplier
      output += doc.getLabel() * alpha * kernel.similarityK(doc.getFeatures(), features);
    }
    return output - bias;
  }

  @Override
  public void trainEpoch() {
    System.out.println("Beginning training... ");
    int numChanged = 0;
    boolean examineAll = true;
    while (numChanged > 0 || examineAll) {
      numChanged = 0;
      if (examineAll)
        for (int i2 = 0; i2 < trainingData.size(); i2++) {
          i2 += examineExample(i2);
          examineAll = false;
        }
      else {
        int[] unboundAlphas = IntStream.range(0, trainingData.size())
            .filter(idx -> (lagrangeMults.get(idx) != 0) && lagrangeMults.get(idx) != lmConstraint)
            .toArray();
        for (int i2 : unboundAlphas)
          numChanged += examineExample(i2);
        if (numChanged == 0)
          examineAll = true;
      }
    }

  }

  /**
   * Used to optimize lagrange multipliers i1 and i2; constraints on i2 are used to optimize.
   * Code from paper was used as reference.
   * @param i1
   * @param i2
   * @return
   */
  private boolean takeStep(int i1, int i2) {
    if (i1 == i2)
      return false;
    double alph1 = lagrangeMults.get(i1);
    double alph2 = lagrangeMults.get(i2);
    int y1 = trainingData.get(i1).getLabel();
    int y2 = trainingData.get(i2).getLabel();
    List<Double> feat1 = trainingData.get(i1).getFeatures();
    List<Double> feat2 = trainingData.get(i2).getFeatures();
    double E1 = errorCache[i1];
    double E2 = errorCache[i2];
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
      if (a2 < L)
        a2 = L;
      else if
          (a2 > H) a2 = H;
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
    double newBias = (b1 + b2) / 2;
    updateErrorCache(i1, i2, a1, a2, newBias);
    lagrangeMults.set(i1, a1);
    lagrangeMults.set(i2, a2);
    bias = newBias;
    System.out.printf("Optimized %d and %d \n", i1, i2);
    return true;
  }

  private void updateErrorCache(int i1, int i2, double a1, double a2, double newBias) {
    double delta = bias - newBias;
    int[] optimizedPair = new int[]{i1, i2};
    for (int i : optimizedPair)
      if (lagrangeMults.get(i) > 0 && lagrangeMults.get(i) < lmConstraint)
        errorCache[i] = 0d;
    Document doc1 = trainingData.get(i1);
    Document doc2 = trainingData.get(i2);
    for (int i = 0; i < errorCache.length; i++) {
      if (i == i1 || i == i2)
        continue;
      Document doci = trainingData.get(i);
      errorCache[i] = errorCache[i] +
          doc1.getLabel() * (a1 - lagrangeMults.get(i1)) * kernel.similarityK(doc1.getFeatures(), doci.getFeatures()) +
          doc2.getLabel() * (a2 - lagrangeMults.get(i2)) * kernel.similarityK(doc2.getFeatures(), doci.getFeatures()) +
          delta;
    }
  }

  private int examineExample(int i2) {
    int y2 = trainingData.get(i2).getLabel();
    double alph2 = lagrangeMults.get(i2);
    double E2 = errorCache[i2];
    double r2 = E2 * y2;
    if ((r2 <= -tolerance  && alph2 < lmConstraint) || (r2 >= tolerance && alph2 > 0)) { // Non-bound and violates KKT
      List<Integer> alphaids = IntStream.range(0, lagrangeMults.size()) //List is chosen here, to shuffle if takestep returns false.
          .filter(idx -> (lagrangeMults.get(idx) != 0 && lagrangeMults.get(idx) != lmConstraint))
          .boxed() //For list
          .sorted(Comparator.comparingDouble(idx -> errorCache[idx])) //Largest Error appears first, minimum last
          .collect(Collectors.toList());
      if (alphaids.size() > 1) { //Approximate stepsize with |E2 - E1|.
        //Heuristic 2: find alpha which maximizes step size.
        int i1 = chooseSecondAlpha(E2, alphaids); //TODO: alphaIds or from all alphas??
        if (takeStep(i1, i2))
          return 1;
      }
      //Loop over all non-zero and non-C alpha, starting at a random point
      Collections.shuffle(alphaids);
      for (int i1 : alphaids)
        if (takeStep(i1, i2))
          return 1;
      //Loop over all possible i1, starting at a random point
      //Shuffle and speed up by removing alphas already checked
      Set<Integer> alreadyChecked = new HashSet<>(alphaids);
      List<Integer> toCheck = IntStream.range(0, lagrangeMults.size())
          .filter(idx -> !alreadyChecked.contains(idx))
          .boxed()
          .collect(Collectors.toList());
      for (int i1 : toCheck) {
        if (takeStep(i1, i2))
          return 1;
      }
    }
    return 0;
  }

  private static int chooseSecondAlpha(double E2, List<Integer> alphaIds) {
    assert(alphaIds.get(0) > alphaIds.get(alphaIds.size() - 1)); //First element should be largest, last smallest.
    return E2 <= 0 ? alphaIds.get(0) : alphaIds.get(alphaIds.size() - 1);
  }

  private double obj(double alpha1, double alpha2, int y1, int y2, double E1, double E2, double k11, double k12, double k22, double bound) {
    int s = y1 * y2;
    double f1 = y1 * (E1 + bias) - alpha1 * k11 - s * alpha2 * k12;
    double f2 = y2 * (E2 + bias) - s * alpha1 * k12 - alpha2 * k22;
    double bound1 = alpha1 + s * (alpha2 - bound);
    return (bound1 * f1) + (bound * f2) + (bound1 * bound1 * k11)/2 + (bound * bound * k22)/2 + (s * bound * bound1 * k12);
  }

}
