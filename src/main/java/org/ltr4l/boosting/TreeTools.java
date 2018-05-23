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

import java.util.*;
import java.util.stream.Collectors;

public class TreeTools {
  private TreeTools(){}
  public static List<Document> orderByFeature(List<Document> documents, int feature){
    assert(feature >= 0 && feature < documents.get(0).getFeatureLength());
    return documents.stream().sorted(Comparator.comparingDouble(doc -> doc.getFeature(feature))).collect(Collectors.toCollection(ArrayList::new));
  }

  /**
   * Finds the best leaf to add a split.
   * @param leafThresholdMap
   * @return
   */
  public static Split findOptimalLeaf(Map<Split, OptimalLeafLoss> leafThresholdMap){
    Iterator<Map.Entry<Split, OptimalLeafLoss>> iterator = leafThresholdMap.entrySet().iterator();
    Map.Entry<Split, OptimalLeafLoss> optimalEntry = iterator.next();
    while(iterator.hasNext()){
      Map.Entry<Split, OptimalLeafLoss> nextEntry = iterator.next();
      if(nextEntry.getValue().getMinLoss() < optimalEntry.getValue().getMinLoss())
        optimalEntry = nextEntry;
    }
    return optimalEntry.getKey();
  }

  /**
   * This method is used to find the threshold and feature which provides the "best" split
   * (the split with the lowest loss). Looks at all features and candidate thresholds.
   * @param leafDocs Docs which landed on a particular leaf.
   * @param numSteps
   * @return
   */
  public static OptimalLeafLoss findMinLeafThreshold(List<Document> leafDocs, int numSteps){ //Faster than default.
    int featureToSplit = 0;
    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(leafDocs, featureToSplit);
    double[] featLoss = findThreshold(sortedDocs, numSteps);
    for(int featId = 1; featId < sortedDocs.getFeatureLength(); featId++){
      double[] error = findThreshold(FeatureSortedDocs.get(leafDocs, featId), numSteps);
      if(error[1] < featLoss[1]){
        featLoss = error;
        featureToSplit = featId;
      }
    }
    double loss = featLoss[1];
    double threshold = featLoss[0];
    return new OptimalLeafLoss(featureToSplit, threshold, loss);
  }

  public static OptimalLeafLoss findMinLeafThreshold(List<Document> leafDocs){ //Note: this can be slow!
    return findMinLeafThreshold(leafDocs, 10);
  }

  /**
   * Finds the best threshold for a given feature.
   * @param fSortedDocs
   * @param numSteps
   * @return
   */
  public static double[] findThreshold(FeatureSortedDocs fSortedDocs, int numSteps){
    if(numSteps <= 1 || numSteps >= fSortedDocs.getFeatureSortedDocs().size())
      return findThreshold(fSortedDocs);
    double fmin = fSortedDocs.getMinFeature();
    double fmax = fSortedDocs.getMaxFeature();
    if(fmax == fmin)
      return new double[]{Double.POSITIVE_INFINITY, Double.POSITIVE_INFINITY}; //Skip this
    double step = Math.abs(fmax - fmin) / numSteps;
    double[] thresholds = new double[numSteps + 1];
    thresholds[0] = fmin;
    for(int i = 1; i <= numSteps; i++)
      thresholds[i] = thresholds[i - 1] + step; //TODO: make last threshold positive infinity?

    List<Document> samples = fSortedDocs.getFeatureSortedDocs();
    double[] featureSamples = fSortedDocs.getFeatureSamples();

    double finalThreshold = fmin;
    double minLoss = Double.POSITIVE_INFINITY;

    for(double threshold : thresholds){
      int idx = binaryThresholdSearch(featureSamples, threshold);
      List<Document> lDocs = new ArrayList<>(samples.subList(0, idx));
      List<Document> rDocs = new ArrayList<>(samples.subList(idx, samples.size()));
      double loss = calcSplitLoss(lDocs) + calcSplitLoss(rDocs);
      if(loss < minLoss){
        finalThreshold = threshold;
        minLoss = loss;
      }
    }
    return new double[] {finalThreshold, minLoss};
  }

  public static double[] findThreshold(FeatureSortedDocs featureSortedDocs){
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
      List<Document> lDocs = new ArrayList<>(fSortedDocs.subList(0, threshId + 1));
      List<Document> rDocs = new ArrayList<>(fSortedDocs.subList(threshId + 1, numDocs));
      double loss = calcSplitLoss(lDocs) + calcSplitLoss(rDocs);
      if(loss < minLoss){
        threshold = !rDocs.isEmpty() ? rDocs.get(0).getFeature(feat) : lDocs.get(lDocs.size() - 1).getFeature(feat);
        minLoss = loss;
      }
    }
    return new double[] {threshold, minLoss};
  }

  public static double calcSplitLoss(List<Document> subData){
    if (subData.size() == 0) return 0;
    double avg = subData.stream().mapToDouble(doc -> doc.getLabel()).sum() / subData.size();
    return subData.stream().mapToDouble(doc -> Math.pow(doc.getLabel() - avg, 2)).sum();
  }

  public static int findMinLossFeat(double[][] thresholds){
    return findMinLossFeat(thresholds, -1);
  }

  public static int findMinLossFeat(double[][] thresholds, double minLoss){
    int feat = -1;
    double loss = Double.POSITIVE_INFINITY;
    for (int fid  = 0; fid < thresholds.length; fid++){
      if(thresholds[fid][1] < loss && thresholds[fid][1] > minLoss){
        loss = thresholds[fid][1];
        feat = fid;
      }
    }
    return feat;
  }

  protected static int binaryThresholdSearch(double[] featSamples, double threshold){
    int lo = 0;
    int hi = featSamples.length;
    int mid = findMid(lo, hi);
    while(lo < hi){
      if(hi - lo == 1){
        return featSamples[lo] >= threshold ? lo : hi;
      }
      if(featSamples[mid] < threshold){
        lo = mid;
        mid = findMid(lo, hi);
      }
      else {
        hi = mid;
        mid = findMid(lo, hi);
      }
    }
    return mid;
  }

  private static int findMid(int lo, int hi){
    int diff = hi - lo;
    int dHalf = diff % 2 == 1 ? (diff / 2) + 1 : diff / 2;
    return lo + dHalf;
  }


}
