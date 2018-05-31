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

import java.util.List;

public class FeatureSortedDocs {
  private final List<Document> featureSortedDocs;
  private final int sortedFeature;
  private final double[] featureSamples;

  public static FeatureSortedDocs get(List<Document> docs, int featureToSort){
    return new FeatureSortedDocs(TreeTools.orderByFeature(docs, featureToSort), featureToSort);
  }

  private FeatureSortedDocs(List<Document> featureSortedDocs, int sortedFeature){
    this.featureSortedDocs = featureSortedDocs;
    this.sortedFeature = sortedFeature;
    this.featureSamples = featureSortedDocs.stream().mapToDouble(doc -> doc.getFeature(sortedFeature)).toArray();
  }

  public int getSortedFeature() {
    return sortedFeature;
  }

  public List<Document> getFeatureSortedDocs() {
    return featureSortedDocs;
  }

  public double getMinFeature(){
    return featureSortedDocs.get(0).getFeature(sortedFeature);
  }

  public double getMaxFeature(){
    int last= featureSortedDocs.size() - 1;
    return featureSortedDocs.get(last).getFeature(sortedFeature);
  }

  public double getFeatureFromIndex(int i){
    return featureSortedDocs.get(i).getFeature(sortedFeature);
  }

  public int getFeatureLength(){
    return featureSortedDocs.get(0).getFeatureLength();
  }

  public double[] getFeatureSamples() {
    return featureSamples;
  }
}
