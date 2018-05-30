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
import java.util.List;

public class RegressionTreeTools extends TreeTools {

  public RegressionTreeTools(){}

  @Override
  public double[] findThreshold(FeatureSortedDocs featureSortedDocs){
    List<Document> fSortedDocs = featureSortedDocs.getFeatureSortedDocs();
    if(featureSortedDocs.getMaxFeature() == featureSortedDocs.getMinFeature())
      return new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}; //Skip this feature
    int feat = featureSortedDocs.getSortedFeature();
    int numDocs = fSortedDocs.size();
    double threshold = fSortedDocs.get(0).getFeature(feat);
    double minLoss = Double.POSITIVE_INFINITY;
    for(int threshId = 0; threshId < numDocs; threshId++ ){
      //Consider cases where sortedFeature values are the same!
      if (threshId != numDocs -1 && fSortedDocs.get(threshId).getFeature(feat) == fSortedDocs.get(threshId + 1).getFeature(feat)) continue;
      List<Document> lDocs = new ArrayList<>(fSortedDocs.subList(0, threshId + 1)); //TODO: check +1
      List<Document> rDocs = new ArrayList<>(fSortedDocs.subList(threshId + 1, numDocs));
      double loss = calcWLloss(lDocs) + calcWLloss(rDocs);
      if(loss < minLoss){
        threshold = !rDocs.isEmpty() ? rDocs.get(0).getFeature(feat) : lDocs.get(lDocs.size() - 1).getFeature(feat);
        minLoss = loss;
      }
    }
    return new double[] {threshold, minLoss};
  }

  @Override
  protected double[] searchStepThresholds(FeatureSortedDocs fSortedDocs, double[] thresholds){
    double[] featureSamples = fSortedDocs.getFeatureSamples();
    List<Document> samples = fSortedDocs.getFeatureSortedDocs();
    double finalThreshold = thresholds[0]; //minimum feature
    double minLoss = Double.POSITIVE_INFINITY;

    for(double threshold : thresholds){
      int idx = binaryThresholdSearch(featureSamples, threshold);
      List<Document> lDocs = samples.subList(0, idx);
      List<Document> rDocs = samples.subList(idx, samples.size());
      double loss = calcWLloss(lDocs) + calcWLloss(rDocs);
      if(loss < minLoss){
        finalThreshold = threshold;
        minLoss = loss;
      }
    }
    return new double[] {finalThreshold, minLoss};
  }

  protected double calcWLloss(List<Document> subData){
    if (subData.size() == 0) return 0;
    double avg = subData.stream().mapToDouble(doc -> doc.getLabel()).sum() / subData.size();
    return subData.stream().mapToDouble(doc -> Math.pow(doc.getLabel() - avg, 2)).sum();
  }
}
