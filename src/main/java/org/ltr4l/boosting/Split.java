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

public class Split { //Node for trees
  private Split source;
  private Split leftLeaf; //TODO: Use List?
  private Split rightLeaf;
  private double threshold;
  private double score;
  private int featureId;
  private final List<Document> scoredDocs;
  private final int leafId;

  //TODO: Add builder for Split.
  protected Split(int featureId, double threshold, List<Document> scoredDocs) throws InvalidFeatureThresholdException { //For root node.
    this.source = null;
    this.featureId = featureId;
    this.threshold = threshold;
    this.scoredDocs = scoredDocs;
    score = 0.0d;
    List<Document> leftDocs = new ArrayList<>();
    List<Document> rightDocs = new ArrayList<>();
    for (Document doc : this.scoredDocs) {
      if (doc.getFeature(featureId) < threshold) leftDocs.add(doc);
      else rightDocs.add(doc);
    }
    leftLeaf = new Split(this, leftDocs, 1);
    rightLeaf = new Split(this, rightDocs, 2);
    leafId = 0;
  }

  protected Split(Split source, List<Document> scoredDocs, int leafId) throws InvalidFeatureThresholdException {
    if (scoredDocs.isEmpty()) throw new InvalidFeatureThresholdException();
    this.source = source;
    this.scoredDocs = scoredDocs;
    this.leafId = leafId;
    leftLeaf = null;
    rightLeaf = null;
    score = 0.0d;
    threshold = Double.NEGATIVE_INFINITY;
    featureId = -1;
  }

  protected Split(Split source, int featureId, double threshold, int leafId, double score){ //Used when reading model!
    this.source = source;
    this.featureId = featureId;
    this.threshold = threshold;
    this.leafId = leafId;
    this.score = score;
    scoredDocs = new ArrayList<>();
  }

  protected void addSplit(int feature, double threshold) throws InvalidFeatureThresholdException {
    this.featureId = feature;
    this.threshold = threshold;
    List<Document> leftDocs = new ArrayList<>();
    List<Document> rightDocs = new ArrayList<>();
    for (Document doc : this.scoredDocs) {
      if (doc.getFeature(featureId) < threshold) leftDocs.add(doc);
      else rightDocs.add(doc);
    }
    leftLeaf = new Split(this, leftDocs, 2 * leafId + 1);
    rightLeaf = new Split(this, rightDocs, (2 * leafId) + 2);
  }

  protected List<Split> getTerminalLeaves() {
    List<Split> terminalLeaves = new ArrayList<>();
    if (!hasDestinations()) {
      terminalLeaves.add(this);
      return terminalLeaves;
    }
    getDestinations().forEach(leaf -> terminalLeaves.addAll(leaf.getTerminalLeaves()));
    return terminalLeaves;
  }

  protected double calculateScore(List<Double> features) {
    assert (leavesProperlySet());
    if (!hasDestinations()) return score;
    Split destination = features.get(featureId) < threshold ? leftLeaf : rightLeaf;
    return destination.calculateScore(features);
  }

  protected void fill(List<Double> info, RegressionTree.DoubleProp type) {
    assert (type != null);
    double prop;
    if (type == RegressionTree.DoubleProp.THRESHOLD)
      prop = threshold;
    else //only two DoubleProps...
      prop = score;
    info.add(prop);
    if (hasDestinations()) {
      for (Split destination : getDestinations()) {
        destination.fill(info, type);
      }
    }
  }

  protected void fill(List<Integer> info, RegressionTree.IntProp type) {
    assert(type != null);
    int prop;
    if (type == RegressionTree.IntProp.FEATURE)
      prop = featureId;
    else
      prop = leafId;
    info.add(prop);
    if (hasDestinations()) {
      for (Split destination : getDestinations()) {
        destination.fill(info, type);
      }
    }
  }

  protected boolean leavesProperlySet() {
    if(leftLeaf == null && rightLeaf == null) return true;
    if(leftLeaf != null && rightLeaf != null) return true;
    return false;
  }

  protected boolean hasDestinations() {
    return !(leftLeaf == null && rightLeaf == null);
  }

  protected List<Split> getDestinations() {
    List<Split> destinations = new ArrayList<>();
    destinations.add(leftLeaf);
    destinations.add(rightLeaf);
    return destinations;
  }

  public List<Document> getScoredDocs() {
    return scoredDocs;
  }
  public void setSource(Split source) {
    this.source = source;
  }
  public Split getSource() { return source; }
  public void setScore(double score) {
    this.score = score;
  }
  public boolean isRoot() {
    return source == null;
  }
  protected void setLeftLeaf(Split leftLeaf) {
    this.leftLeaf = leftLeaf;
  }
  protected void setRightLeaf(Split rightLeaf) {
    this.rightLeaf = rightLeaf;
  }
  protected Split getLeftLeaf() { return leftLeaf; }
  protected Split getRightLeaf() { return rightLeaf; }
  public int getLeafId() { return leafId; }
  public double getThreshold() { return threshold; }
  public double getScore() { return score; }
  public int getFeatureId() { return featureId; }
}
