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

  public static RegressionTree.Split findOptimalLeaf(Map<RegressionTree.Split, double[]> leafThresholdMap){
    Iterator<Map.Entry<RegressionTree.Split, double[]>> iterator = leafThresholdMap.entrySet().iterator();
    Map.Entry<RegressionTree.Split, double[]> optimalEntry = iterator.next();
    while(iterator.hasNext()){
      Map.Entry<RegressionTree.Split, double[]> nextEntry = iterator.next();
      if(nextEntry.getValue()[1] < optimalEntry.getValue()[1])
        optimalEntry = nextEntry;
    }
    return optimalEntry.getKey();
  }

  public static double[] findMinThreshold(RegressionTree.Split leaf){ //TODO: Faster code
    int featureToSplit = 0;
    List<Document> sortedDocs = orderByFeature(leaf.getScoredDocs(), 0);
    double[] featLoss = findThreshold(sortedDocs, 0);
    for(int featId = 1; featId < sortedDocs.get(0).getFeatureLength(); featId++){
      double[] error = findThreshold(orderByFeature(sortedDocs, featId), featId);
      if(error[1] < featLoss[1]){
        featLoss = error;
        featureToSplit = featId;
      }
    }
    double loss = featLoss[1];
    double threshold = featLoss[0];
    return new double[] {featureToSplit, loss, threshold};
  }

  public static double[] findThreshold(List<Document> fSortedDocs, int feat){
    int numDocs = fSortedDocs.size();
    double threshold = fSortedDocs.get(0).getFeature(feat);
    double minLoss = Double.POSITIVE_INFINITY;
    for(int threshId = 0; threshId < numDocs; threshId++ ){
      //Consider cases where feature values are the same!
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
}
