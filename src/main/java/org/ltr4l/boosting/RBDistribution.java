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
package org.ltr4l.boosting;

import org.ltr4l.query.RankedDocs;

import java.util.List;

public class RBDistribution {
  private final double[][][] dist;
  private double normFactor;


  public static RBDistribution getInitDist(List<RankedDocs> rQueries, int correctPairNum){
    RBDistribution iDist = new RBDistribution(rQueries.size());
    iDist.initialize(rQueries, correctPairNum);
    return iDist;
  }


  protected RBDistribution(int queryNum){
    dist = new double[queryNum][][];
    normFactor = 1d; //normFactor will change depending on iteration of the distribution. Initially, it should be 1.
  }

  private void initialize(List<RankedDocs> rQueries, int correctPairNum){
    for(int i = 0; i < rQueries.size(); i++)
      dist[i] = initQueryDist(rQueries.get(i), correctPairNum);
  }

  private static double[][] initQueryDist(RankedDocs rDocs, int correctPairNum){
    assert(rDocs.size() >= 2);
    int size = rDocs.size();
    double[][] D_0 = new double[size][size];
    for(int i = 0; i < size - 1; i++){
      if(rDocs.getLabel(i) == 0) return D_0; //Speedup. As the list is ranked, there should be no valid pairs after 0.
      for(int j = i + 1; j < size; j++){
        if(rDocs.getLabel(i) > rDocs.getLabel(j))
          D_0[i][j] = 1d / correctPairNum;
      }
    }
    return D_0;
  }

  public RBDistribution update(WeakLearner wl, List<RankedDocs> queries ){ //TODO: Restrict Ranker type to RankBoost?
    double newNormFactor = 0d;
    for(int qid = 0; qid < queries.size(); qid++){
      newNormFactor += updateQuery(wl, qid, queries.get(qid));
    }
    normalize(newNormFactor);
    return this;
  }

  protected double updateQuery(WeakLearner wl, int qid, RankedDocs rankedDocs){ //returns the query normalization factor
    double newNormFactor = 0d;
    for(int i = 0; i < rankedDocs.size() - 1; i++){
      for(int j = rankedDocs.size() - 1; j >= i + 1; j--){ //Speedup. Go back until equivalent label is reached.
        if(dist[qid][i][j] == 0) continue;
        dist[qid][i][j] *= Math.exp(wl.getAlpha() * (wl.predict(rankedDocs.get(i).getFeatures()) - wl.predict(rankedDocs.get(j).getFeatures())));
        newNormFactor += dist[qid][i][j];
      }
    }
    return newNormFactor;
  }

  private void normalize(double normFactor){
    for(int qid = 0; qid < dist.length; qid++){
      for(int i = 0; i < dist[qid].length; i++)
        for(int j = 0; j < dist[qid][i].length; j++) //Note: could be sped up
          dist[qid][i][j] /= normFactor;
    }
    this.normFactor = normFactor;
  }

  public double[][][] getFullDist(){
    return dist;
  }

  public double[][] getQueryDist(int i){
    return dist[i];
  }

  public void setQueryDist(int i, double[][] newDist){
    dist[i] = newDist;
  }

  public double getNormFactor() {
    return normFactor;
  }
}
