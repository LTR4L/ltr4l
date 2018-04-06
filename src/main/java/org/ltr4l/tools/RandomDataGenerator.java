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

package org.ltr4l.tools;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

public class RandomDataGenerator {

  private static final Random r = new Random(System.currentTimeMillis());

  /*
   * 1. Generate (s-1) formula (linear model) y[k] = wx + b[k] (k=1,2,...,s-1) where w and b[1]
   *    are random values, s is the number of stars, and b[i] = b[i-1] + A (A>0)
   * 2. Generate set of sample data X. Pick x ∈ X. Calculate y[k]. If y[1] > 0, then
   *    let the data x have label "star-1", else if y[2] > 0, then let the data x have label
   *    "star-2", else if y[3] > 0, then let the data x have label "star-3", and so on.
   * 3. Repeat 1 and 2 if you want more Query.
   */
  public static void main(String[] args) throws Exception {

    final int numStars = 4;
    final int dim = 4;
    final double alpha = alpha(dim, numStars);
    final double b1 = 0;

    // generate boundaries
    List<LinearModel> linearModels = new ArrayList<>();
    LinearModel y = LinearModel.generate(dim, b1);
    System.out.println(y.toString());
    linearModels.add(y);
    for(int i = 1; i < numStars - 1; i++){
      y = y.nextBoundary(alpha);
      System.out.println(y.toString());
      linearModels.add(y);
    }

    // generate data
    final int numData = 100;
    for(int i = 0; i < numData; i++){
      double[] sample = generateSample(dim);
      int star;
      for(star = 0; star < numStars - 1; star++){
        if(linearModels.get(star).predict(sample)){
          break;
        }
      }
      System.out.printf("%f => %d\n", sample[0], star);
    }
  }

  private static final int MAX_TRY = 30;  // TODO: should be taken into account?
  private final int dim;
  private final int numStars;
  private final double alpha;
  private List<LinearModel> linearModels;

  public RandomDataGenerator(int dim, int numStars){
    this.dim = dim;
    this.numStars = numStars;
    this.alpha  = alpha(dim, numStars);
    this.linearModels = generateBoundaries(dim, numStars, alpha);
  }

  public static List<LinearModel> generateBoundaries(int dim, int numStars, double alpha){
    List<LinearModel> linearModels = new ArrayList<>();
    final double b1 = 0;
    LinearModel y = LinearModel.generate(dim, b1);
    System.out.println(y.toString());
    linearModels.add(y);
    for(int i = 1; i < numStars - 1; i++){
      y = y.nextBoundary(alpha);
      System.out.println(y.toString());
      linearModels.add(y);
    }

    return linearModels;
  }

  public QuerySet getRandomQuerySet(int numQueries, int numSamplesPerQuery, int minSamples){
    QuerySet querySet = new QuerySet();

    int qId = 0;
    int triedPerModel = 0;
    while(qId < numQueries){
      Query query = generateQuery(qId, numSamplesPerQuery);

      // check if it satisfy the condition with the number of samples...
      Map<Integer, Integer> map = new HashMap<>();
      for(Document doc: query.getDocList()){
        Integer count = map.get(doc.getLabel());
        if(count == null){
          map.put(doc.getLabel(), 1);
        }
        else{
          map.put(doc.getLabel(), count + 1);
        }
      }

      boolean satisfied = true;
      for(int i = 0; i < numStars; i++){
        Integer count = map.get(i);
        if(count == null || count < minSamples){
          satisfied = false;
          System.err.printf("WARNING: The number of samples for label \"%d\" is less than %d. Will try again\n", i, minSamples);
          triedPerModel++;
          break;
        }
      }
      if(satisfied){
        querySet.addQuery(query);
        qId++;
      }
      else if(triedPerModel == MAX_TRY){
        System.err.println("WARNING: The built model is not good. Will create another one.");
        linearModels = generateBoundaries(dim, numStars, alpha);
        querySet = new QuerySet();
        triedPerModel = 0;
        qId = 0;
      }
    }

    return querySet;
  }

  public static double alpha(int dim, int stars){
    assert(stars >= 2);
    return (0.2 - 0.05 * (dim - 1)) / Math.pow(2, stars - 2);
  }

  public static double[] generateSample(int d){
    double[] sample = new double[d];
    for(int i = 0; i < d; i++){
      sample[i] = r.nextDouble() - 0.5;
    }
    return sample;
  }

  public Query generateQuery(int qId, int numSamplesPerQuery){
    Query query = new Query();
    for(int i = 0; i < numSamplesPerQuery; i++){
      query.addDocument(generateDocument());
    }

    query.setQueryId(qId);
    return query;
  }

  public Document generateDocument(){
    double[] sample = generateSample(dim);
    Document document = new Document();
    for(double d: sample){
      document.addFeature(d);
    }

    int star;
    for(star = 0; star < numStars - 1; star++){
      if(linearModels.get(star).predict(sample)){
        break;
      }
    }

    document.setLabel(star);

    return document;
  }

  public static class LinearModel {

    private final int d;
    private final double[] weights;
    private final double b;

    public static LinearModel generate(int d, double b){
      LinearModel linearModel = new LinearModel(d, b);
      setRandomWeights(linearModel.weights);
      return linearModel;
    }

    private LinearModel(int d, double b){
      assert(d > 0);
      this.d = d;
      this.b = b;
      weights = new double[d];
    }

    private LinearModel(double[] weights, double b){
      assert(weights.length > 0);
      this.d = weights.length;
      this.weights = weights;
      this.b = b;
    }

    public LinearModel nextBoundary(double alpha){
      return new LinearModel(weights, b + alpha);
    }

    public boolean predict(double[] x){
      assert(weights.length == x.length);
      double y = 0;
      for( int i = 0; i < weights.length; i++){
        y += weights[i] * x[i];
      }
      y += b;

      return y > 0 ? true : false;
    }

    @Override
    public String toString(){
      StringBuilder sb = new StringBuilder();
      sb.append("y = ");
      sb.append('(').append(weights[0]).append(")*x1");
      for(int i = 1; i < d; i++){
        sb.append(" + (").append(weights[i]).append(")*x").append(i + 1);
      }
      sb.append(" + (").append(b).append(')');
      return sb.toString();
    }

    static void setRandomWeights(double[] weights){
      for(int i = 0; i < weights.length; i++){
        weights[i] = r.nextDouble() - 0.5;
      }
    }
  }
}
