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

  public static RegressionTree.Split findOptimalLeaf(Map<RegressionTree.Split, OptimalLeafLoss> leafThresholdMap){
    Iterator<Map.Entry<RegressionTree.Split, OptimalLeafLoss>> iterator = leafThresholdMap.entrySet().iterator();
    OptimalLeafLoss optimalValue = iterator.next().getValue();
    while(iterator.hasNext()){
      OptimalLeafLoss nextValue = iterator.next().getValue();
      if(nextValue.getMinLoss() < optimalValue.getMinLoss())
        optimalValue = nextValue;
    }
    return optimalValue.getLeaf();
  }

  public static OptimalLeafLoss findMinThreshold(RegressionTree.Split leaf, int numSteps){ //Faster than default.
    int featureToSplit = 0;
    List<Document> leafDocs = leaf.getScoredDocs();
    if(numSteps > leafDocs.size()) numSteps = 0; //Just consider all candidate features if numSteps is greater.
    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(leafDocs, numSteps);
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
    return new OptimalLeafLoss(leaf, featureToSplit, threshold, loss);
  }

  public static OptimalLeafLoss findMinThreshold(RegressionTree.Split leaf){ //Note: this can be slow!
    return findMinThreshold(leaf, 10);
  }

  //For using a statistical method to find thresholds, rather than checking all possible values.
  //This will be used for speedup.
  public static double[] findThreshold(FeatureSortedDocs fSortedDocs, int numSteps){
    if(numSteps <= 0) return findThreshold(fSortedDocs);
    double fmin = fSortedDocs.getMinFeature();
    double fmax = fSortedDocs.getMaxFeature();
    double step = Math.abs(fmax - fmin) / numSteps;
    double[] thresholds = new double[numSteps];
    thresholds[0] = fmin;
    for(int i = 1; i < numSteps; i++)
      thresholds[i] = thresholds[i - 1] + step;

    List<Document> samples = fSortedDocs.getFeatureSortedDocs();
    int feat = fSortedDocs.getSortedFeature();
    double[] featureSamples = fSortedDocs.getFeatureSamples();

    double finalThreshold = fmin;
    double minLoss = Double.POSITIVE_INFINITY;

    for(double threshold : thresholds){
      int idx = binaryThresholdSearch(featureSamples, threshold);
      List<Document> lDocs = new ArrayList<>(samples.subList(0, idx));
      List<Document> rDocs = new ArrayList<>(samples.subList(idx, samples.size()));
      double loss = calcThresholdLoss(lDocs) + calcThresholdLoss(rDocs);
      if(loss < minLoss){
        finalThreshold = !rDocs.isEmpty() ? rDocs.get(0).getFeature(feat) : lDocs.get(lDocs.size() - 1).getFeature(feat);
        minLoss = loss;
      }
    }
    return new double[] {finalThreshold, minLoss};
  }

  public static double[] findThreshold(FeatureSortedDocs featureSortedDocs){
    List<Document> fSortedDocs = featureSortedDocs.getFeatureSortedDocs();
    int feat = featureSortedDocs.getSortedFeature();

    int numDocs = fSortedDocs.size();
    double threshold = fSortedDocs.get(0).getFeature(feat);
    double minLoss = Double.POSITIVE_INFINITY;
    for(int threshId = 0; threshId < numDocs; threshId++ ){
      //Consider cases where sortedFeature values are the same!
      if (threshId != numDocs -1 && fSortedDocs.get(threshId).getFeature(feat) == fSortedDocs.get(threshId + 1).getFeature(feat)) continue;
      List<Document> lDocs = new ArrayList<>(fSortedDocs.subList(0, threshId + 1));
      List<Document> rDocs = new ArrayList<>(fSortedDocs.subList(threshId + 1, numDocs));
      double loss = calcThresholdLoss(lDocs) + calcThresholdLoss(rDocs);
      if(loss < minLoss){
        threshold = !rDocs.isEmpty() ? rDocs.get(0).getFeature(feat) : lDocs.get(lDocs.size() - 1).getFeature(feat);
        minLoss = loss;
      }
    }
    return new double[] {threshold, minLoss};
  }

  public static double calcThresholdLoss(List<Document> subData){
    if (subData.size() == 0) return 0;
    double avg = subData.stream().mapToDouble(doc -> doc.getLabel()).sum() / subData.size();
    return subData.stream().mapToDouble(doc -> Math.pow(doc.getLabel() - avg, 2)).sum();
  }

  public static int findMinLossFeat(double[][] thresholds){
    return findMinLossFeat(thresholds, 0.0d);
  }

  //TODO: Faster code?
  public static int findMinLossFeat(double[][] thresholds, double minLoss){
    int feat = 0;
    double loss = Double.POSITIVE_INFINITY;
    for (int fid  = 0; fid < thresholds.length; fid++){
      if(thresholds[fid][1] < loss && thresholds[fid][1] > minLoss){
        loss = thresholds[feat][1];
        feat = fid;
      }
    }
    return feat;
  }

  //Initialize with initial midpoint.
  private static int binaryThresholdSearch(double[] featSamples, double threshold){
    int lo = 0;
    int hi = featSamples.length;
    int mid = findMid(lo, hi);
    while(lo < hi){
      if(hi - lo == 1) return hi;
      if(featSamples[mid] <= threshold){
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
