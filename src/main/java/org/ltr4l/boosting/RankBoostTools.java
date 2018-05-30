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

import java.util.*;

public class RankBoostTools extends TreeTools {
  private final double[][] potential; //Since all threshold calculations require potential...
  private final Map<Document, int[]> docMap; //{qid, index}, since FeatureSortedDocs are used...

  public RankBoostTools(double[][] potential, List<RankedDocs> queries){
    this.potential = Objects.requireNonNull(potential);
    Objects.requireNonNull(queries);
    this.docMap = new HashMap<>();
    for(int qid = 0; qid < queries.size(); qid++){
      RankedDocs query = queries.get(qid);
      for(int idx = 0; idx < query.size(); idx++)
        docMap.put(query.get(idx), new int[] {qid, idx});
    }
  }

  @Override
  public double[] findThreshold(FeatureSortedDocs featureSortedDocs){
    List<Document> samples = featureSortedDocs.getFeatureSortedDocs();
    if(featureSortedDocs.getMaxFeature() == featureSortedDocs.getMinFeature())
      return new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}; //Skip this feature
    int feat = featureSortedDocs.getSortedFeature();
    int numDocs = samples.size();
    double threshold = samples.get(0).getFeature(feat);
    double maxr = Double.NEGATIVE_INFINITY;
    int qdef = 0;
    for(int threshIdx = 0; threshIdx < numDocs; threshIdx++ ){
      //Consider cases where sortedFeature values are the same!
      if (threshIdx != numDocs -1 && samples.get(threshIdx).getFeature(feat) == samples.get(threshIdx + 1).getFeature(feat)) continue;
      double R = Arrays.stream(potential).mapToDouble(q -> Arrays.stream(q).sum()).sum();
      double L = calcWLloss(samples.subList(threshIdx, numDocs));
      int q = Math.abs(L) > Math.abs(L - R) ? 0 : 1;
      double r = Math.abs(L - (q * R));
      if(r > maxr){
        threshold = featureSortedDocs.getFeatureFromIndex(threshIdx);
        maxr = r;
        qdef = q;
      }
    }
    return new double[] {threshold, 1/maxr, qdef};
  }

  @Override
  protected double[] searchStepThresholds(FeatureSortedDocs fSortedDocs, double[] thresholds) { //find the threshold which maximizes r, and return error Z
    double[] featureSamples = fSortedDocs.getFeatureSamples();
    List<Document> samples = fSortedDocs.getFeatureSortedDocs();
    double finalThreshold = thresholds[0];
    double maxr = Double.NEGATIVE_INFINITY;
    int qdef = 0;
    for(double threshold : thresholds){
      double R = Arrays.stream(potential).mapToDouble(q -> Arrays.stream(q).sum()).sum(); //See pseudo code in original paper
      int idx = binaryThresholdSearch(featureSamples, threshold);
      double L = calcWLloss(samples.subList(idx, samples.size()));
      int q = Math.abs(L) > Math.abs(L - R) ? 0 : 1;
      double r = Math.abs(L - (q * R));
      if(r > maxr){
        finalThreshold = threshold;
        maxr = r;
        qdef = q;
      }
    }
    return new double[] {finalThreshold, 1/maxr, qdef}; //Note 1/maxr is returned as the minimum is searched for in parent.
  }

  protected double calcWLloss(List<Document> subData){
    if (subData.size() == 0) return 0;
    double L = 0d;
    for(Document doc : subData){
      int qid = docMap.get(doc)[0];
      int index = docMap.get(doc)[1];
      L += potential[qid][index];
    }
    return L;
  }

}
