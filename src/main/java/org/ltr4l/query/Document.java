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

package org.ltr4l.query;

import java.util.ArrayList;
import java.util.List;

public class Document {
  private int label;
  private final List<Double> features;

  public Document() {
    features = new ArrayList<>();
  }
  public Document(List<Double> features, int label) {
    this.features = features;
    this.label = label;
  }

  public int getLabel() {
    return label;
  }

  public void setLabel(int newLabel) {
    label = newLabel;
  }

  public List<Double> getFeatures() {
    return features;
  }

  public double getFeature(int i){
    return features.get(i);
  }

  public int getFeatureLength(){
    return features.size();
  }

  public void addFeature(double feature) {
    features.add(feature);
  }
}
