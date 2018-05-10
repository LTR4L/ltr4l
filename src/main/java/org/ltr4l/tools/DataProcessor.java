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

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public final class DataProcessor {

  private DataProcessor(){}

  public static List<Document> makeDocList(List<Query> dataSet){
    //List<Document> docList = new ArrayList<>(dataSet.stream().map(query -> query.getDocList()).flatMap(docs -> docs.stream()).collect(Collectors.toList()));
    List<Document> docList = new ArrayList<>();
    for (Query query : dataSet)
      docList.addAll(query.getDocList());
    return docList;
  }

  /**
   * Calculates the variance for each feature in the dataset (List of Documents).
   * Object returned is a map returned with the key being the feature index (Integer) and the value
   * being the variance (double). Note that as the parameter is a list of documents, the variance can be calculated
   * per query or for the whole dataset.
   * @param data
   * @return
   */
  public static HashMap<Integer, Double> calcVariances(List<Document> data){
    assert(data.get(0).getFeatures().size() > 0);
    HashMap variance = new HashMap();
    int featLength = data.get(0).getFeatures().size();
    for(int feature = 0; feature < featLength; feature++){
      int i = feature; //For lambda
      double featAvg = getAvgOfFeature(data, feature);
      double var = data.stream().mapToDouble(doc -> Math.pow(doc.getFeature(i) - featAvg, 2)).sum() / data.size();
      variance.put(feature, var);
    }
    return variance;
  }

  public static double getAvgOfFeature(List<Document> data, int feature){
    assert(feature < data.get(0).getFeatures().size());
    return data.stream().mapToDouble(doc -> doc.getFeature(feature)).sum() / data.size();
  }

  public static List<Integer> orderSelectedFeatures(Map<Integer, Double> variance, double allowance) {
    Stream<Integer> featStream = variance.keySet().stream();
    if (allowance > 0)
      return featStream
          .filter(feat -> variance.get(feat) > allowance)
          .sorted((feat1, feat2) -> Double.compare(variance.get(feat2), variance.get(feat1)))
          .collect(Collectors.toCollection(ArrayList::new));

    return featStream //TODO: Remove if statement and this line...? Will perform filter every time...
        .sorted((feat1, feat2) -> Double.compare(variance.get(feat2), variance.get(feat1)))
        .collect(Collectors.toCollection(ArrayList::new));
  }

  public static List<Document> scale(List<Document> data){
    double[][] featMinMax = getFeatureMinMax(data);
    int featLength = data.get(0).getFeatures().size();
    for(Document doc : data){
      List<Double> features = doc.getFeatures();
      for(int i = 0; i < featLength; i++){
        double feature = features.get(i);
        double min = featMinMax[i][0];
        double max = featMinMax[i][1];
        feature = (feature - min)/ (max - min);
        features.set(i, feature);
      }
    }
    return data;
  }

  public static double[][] getFeatureMinMax(List<Document> data){
    int featLength = data.get(0).getFeatures().size();
    double[][] featMinMax = new double[featLength][2];
    for(int feat = 0; feat < featLength; feat++){
      int i = feat;
      Set<Double> featureSet = data.stream().map(doc -> doc.getFeature(i)).collect(Collectors.toCollection(HashSet::new));
      double min = Collections.min(featureSet);
      double max = Collections.max(featureSet);
      featMinMax[feat] = new double[] {min, max};
    }
    return featMinMax;
  }

  //TODO: Implement method to return data set with selected features only...?
  //TODO: Implement centering
}
