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

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;

import java.io.IOException;
import java.io.Writer;
import java.util.*;

public class RegressionTree extends Ranker<TreeEnsemble.TreeConfig>{
  private final Split root;

  public RegressionTree(int numLeaves, int initFeat, double initThreshold, List<Document> docs) throws InvalidFeatureThresholdException{
    assert(numLeaves >= 2);
    root = new Split(null, initFeat, initThreshold, docs);
    Map<Split, double[]> splitErrorMap = new HashMap<>();
    for(int l = 2; l < numLeaves; l++) {
      for (Split leaf : root.getTerminalLeaves()) {
        splitErrorMap.put(leaf, TreeTools.findMinThreshold(leaf));
      }
      Split optimalLeaf = TreeTools.findOptimalLeaf(splitErrorMap);
      int feature = (int) splitErrorMap.get(optimalLeaf)[0];
      double threshold = splitErrorMap.get(optimalLeaf)[2];
      optimalLeaf.addSplit(feature, threshold);
      splitErrorMap.remove(optimalLeaf);
    }
  }

  public List<Split> getTerminalLeaves(){
    return root.getTerminalLeaves();
  }

  @Override
  public double predict(List<Double> features) {
    return root.calculateScore(features);
  }

  @Override
  public void writeModel(TreeEnsemble.TreeConfig config, Writer writer) throws IOException {
    //TODO: Implement
  }

  public static class Split { //Node for trees
    private Split source;
    private Split leftLeaf; //TODO: Use List?
    private Split rightLeaf;
    private double threshold;
    private double score;
    private int featureId;
    private final List<Document> scoredDocs;

    public Split(Split source, int featureId, double threshold, List<Document> scoredDocs) throws InvalidFeatureThresholdException{ //For root node.
      this.source = source;
      this.featureId = featureId;
      this.threshold = threshold;
      this.scoredDocs = scoredDocs;
      score = 0.0d;
      List<Document> leftDocs = new ArrayList<>();
      List<Document> rightDocs = new ArrayList<>();
      for(Document doc : this.scoredDocs){
        if(doc.getFeature(featureId) < threshold) leftDocs.add(doc);
        else rightDocs.add(doc);
      }
      leftLeaf = new Split(this, leftDocs);
      rightLeaf = new Split(this, rightDocs);
    }

    public Split(Split source, List<Document> scoredDocs) throws InvalidFeatureThresholdException{
      if(scoredDocs.isEmpty()) throw new InvalidFeatureThresholdException();
      this.source = source;
      this.scoredDocs = scoredDocs;
      leftLeaf = null;
      rightLeaf = null;
      score = 0.0d;
      threshold = Double.NEGATIVE_INFINITY;
      featureId = -1;
    }

    public void addSplit(int feature, double threshold) throws InvalidFeatureThresholdException{
      this.featureId = feature;
      this.threshold = threshold;
      List<Document> leftDocs = new ArrayList<>();
      List<Document> rightDocs = new ArrayList<>();
      for(Document doc : this.scoredDocs){
        if(doc.getFeature(featureId) < threshold) leftDocs.add(doc);
        else rightDocs.add(doc);
      }
      leftLeaf = new Split(this, leftDocs);
      rightLeaf = new Split(this, rightDocs);
    }

    public List<Split> getTerminalLeaves(){
      List<Split> terminalLeaves = new ArrayList<>();
      if (!hasDestinations()){
        terminalLeaves.add(this);
        return terminalLeaves;
      }
      getDestinations().forEach(leaf -> terminalLeaves.addAll(leaf.getTerminalLeaves()));
      return terminalLeaves;
    }

    public double calculateScore(List<Double> features){
      assert(leavesProperlySet());
      if(!hasDestinations()) return score;
      Split destination = features.get(featureId) < threshold ? leftLeaf : rightLeaf;
      return destination.calculateScore(features);
    }

    protected boolean leavesProperlySet(){
      if(leftLeaf == null && rightLeaf == null) return true;
      if(leftLeaf != null && rightLeaf != null) return true;
      return false;
    }

    protected boolean hasDestinations(){
      return !(leftLeaf == null && rightLeaf == null);
    }

    protected List<Split> getDestinations(){
      List<Split> destinations = new ArrayList<>();
      destinations.add(leftLeaf);
      destinations.add(rightLeaf);
      return destinations;
    }

    public void setLeftLeaf(Split leftLeaf) { //TODO: Refractor
      assert(leftLeaf == null || this.getRoot() != leftLeaf.getRoot());
      if(this.leftLeaf != null)  this.leftLeaf.setSource(null); //Separate current destination reference to this leaf
      if(leftLeaf != null) leftLeaf.setSource(this);
      this.leftLeaf = leftLeaf;
    }

    public void setRightLeaf(Split rightLeaf) {
      assert(rightLeaf == null || this.getRoot() != rightLeaf.getRoot());
      if(this.rightLeaf != null)  this.rightLeaf.setSource(null); //Separate current destination reference to this leaf
      if(rightLeaf != null) rightLeaf.setSource(this);
      this.rightLeaf = rightLeaf;
    }

    public List<Document> getScoredDocs() {
      return scoredDocs;
    }

    public void setFeatureId(int featureId) {
      this.featureId = featureId;
    }

    public void setSource(Split source) {
      this.source = source;
    }

    public void setScore(double score) {
      this.score = score;
    }

    public Split getRoot(){
      return this.isRoot() ? this : source.getRoot();
    }

    public Split getLeftLeaf(){
      return leftLeaf;
    }

    public Split getRightLeaf(){
      return rightLeaf;
    }

    public boolean isRoot(){
      return source == null;
    }

    public Split getSource() {
      return source;
    }

    protected boolean isLinkedTo(Split leaf){ //Not currently used!
      if(source == leaf) return true;
      if(source.isRoot()) return false;
      return source.isLinkedTo(leaf);
    }
  }

}
