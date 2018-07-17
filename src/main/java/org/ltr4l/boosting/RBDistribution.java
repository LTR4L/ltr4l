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

import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;
import org.ltr4l.trainers.RankBoostTrainer;

import java.util.List;

public class RBDistribution extends Distribution.D3R {

  private static int getCorrectPairNumber(RankedDocs docs){
    int correctPairs = 0;
    for(int i = 0; i < docs.size() - 1; i++){
      if(docs.getLabel(i) == 0) return correctPairs;
      for(int j = docs.size() - 1; j >= i + 1 && docs.getLabel(i) > docs.getLabel(j); j--){ //Go backwards until first label match
        correctPairs++;
      }
    }
    return correctPairs;
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

  public RBDistribution(List<RankedDocs> rQueries){
    super(rQueries);
  }

  protected double[][] calcPotential(){
    double[][] potential = new double[dist.length][];
    for(int qid = 0; qid < dist.length; qid++){
      double[][] qDist = getQueryDist(qid);
      potential[qid] = new double[qDist.length];
      for(int i = 0; i < qDist.length; i++) {  //Note: because of how RBDistribution is created, this can be sped up.
        for(int j = 0; j < qDist.length; j++)
          potential[qid][i] += qDist[j][i] - qDist[i][j];
      }
    }
    return potential;
  }

  protected void initialize(List<RankedDocs> rQueries){
    int correctPairNum = 0;
    for(RankedDocs query : rQueries)
      correctPairNum += getCorrectPairNumber(query);
    for(int i = 0; i < rQueries.size(); i++)
      dist[i] = initQueryDist(rQueries.get(i), correctPairNum);
  }

  @Override
  protected double updateQuery(WeakLearner wl, int qid, List<Document> rankedDocs) {
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

  protected double updateQuery(WeakLearner wl, int qid, RankedDocs rankedDocs){ //returns the query normalization factor
    return updateQuery(wl, qid, rankedDocs.getRankedDocs());
  }
}
