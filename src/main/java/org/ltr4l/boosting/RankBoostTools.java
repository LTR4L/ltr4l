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

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class RankBoostTools extends TreeTools {
  private final double[][] potential; //Since all threshold calculations require potential...
  private final Map<Document, int[]> docMap; //{qid, index}

  public RankBoostTools(double[][] potential, Map<Document, int[]> docMap){
    this.potential = potential;
    this.docMap = docMap;
  }

  @Override
  public double[] findThreshold(FeatureSortedDocs featureSortedDocs){
    List<Document> fSortedDocs = featureSortedDocs.getFeatureSortedDocs();
    if(featureSortedDocs.getMaxFeature() == featureSortedDocs.getMinFeature())
      return new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}; //Skip this feature
    int feat = featureSortedDocs.getSortedFeature();
    int numDocs = fSortedDocs.size();
    double threshold = fSortedDocs.get(0).getFeature(feat);
    double maxr = Double.NEGATIVE_INFINITY;
    int qdef = 0;
    for(int threshId = 0; threshId < numDocs; threshId++ ){
      //Consider cases where sortedFeature values are the same!
      if (threshId != numDocs -1 && fSortedDocs.get(threshId).getFeature(feat) == fSortedDocs.get(threshId + 1).getFeature(feat)) continue;
      double R = Arrays.stream(potential).mapToDouble(q -> Arrays.stream(q).sum()).sum();
      double L = 0d;
      for(int i = threshId; i < fSortedDocs.size(); i++){
        Document doc = fSortedDocs.get(i);
        int qid = docMap.get(doc)[0];
        int index = docMap.get(doc)[1];
        L += potential[qid][index];
      }
      int q = Math.abs(L) > Math.abs(L - R) ? 0 : 1;
      double r = Math.abs(L - (q * R));
      if(r > maxr){
        threshold = threshId;
        maxr = r;
        qdef = q;
      }
    }
    return new double[] {threshold, 1/maxr, qdef};
  }

  @Override
  public double[] searchThresholds(FeatureSortedDocs fSortedDocs, double[] thresholds) { //find the threshold which maximizes r, and return error Z
    double[] featureSamples = fSortedDocs.getFeatureSamples();
    List<Document> samples = fSortedDocs.getFeatureSortedDocs();
    double finalThreshold = thresholds[0];
    double maxr = Double.NEGATIVE_INFINITY;
    int qdef = 0;

    for(double threshold : thresholds){
      double R = Arrays.stream(potential).mapToDouble(q -> Arrays.stream(q).sum()).sum(); //See pseudo code in original paper
      double L = 0d;
      int idx = binaryThresholdSearch(featureSamples, threshold);
      for(int i = idx; i < samples.size(); i++){
        Document doc = samples.get(i);
        int qid = docMap.get(doc)[0];
        int index = docMap.get(doc)[1];
        L += potential[qid][index];
      }
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

  public double calcWLloss(List<Document> subData){
    if (subData.size() == 0) return 0;
    double avg = subData.stream().mapToDouble(doc -> doc.getLabel()).sum() / subData.size();
    return subData.stream().mapToDouble(doc -> Math.pow(doc.getLabel() - avg, 2)).sum();
  }

}
