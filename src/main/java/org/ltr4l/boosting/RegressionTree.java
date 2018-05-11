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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;
import org.ltr4l.query.Document;

import java.io.IOException;
import java.io.Writer;
import java.util.*;

public class RegressionTree extends Ranker<Ensemble.TreeConfig>{
  private final Split root;
  private final int numLeaves;

  public RegressionTree(int numLeaves, int initFeat, double initThreshold, List<Document> docs) throws InvalidFeatureThresholdException{
    assert(numLeaves >= 2);
    this.numLeaves = numLeaves;
    root = new Split(initFeat, initThreshold, docs);
    Map<Split, double[]> splitErrorMap = new HashMap<>();
    for(int l = 2; l < numLeaves; l++) {
      for (Split leaf : root.getTerminalLeaves()) {
        splitErrorMap.put(leaf, TreeTools.findMinThreshold(leaf));
      }
      Split optimalLeaf = TreeTools.findOptimalLeaf(splitErrorMap);
      int feature = (int) splitErrorMap.get(optimalLeaf)[0]; //TODO: feature and threshold should not be in same array.
      double threshold = splitErrorMap.get(optimalLeaf)[2];
      optimalLeaf.addSplit(feature, threshold);
      splitErrorMap.remove(optimalLeaf);
    }
  }

  protected List<Double> getModelInfo(DoubleProp type){
    List<Double> info = new ArrayList<>(); //2 * num leaves - 1 should be final list
    root.fill(info, type); //Start at 0.
    return info;
  }

  protected List<Integer> getModelInfo(IntProp type){
    List<Integer> info = new ArrayList<>();
    root.fill(info, type);
    return info;
  }


  public List<Split> getTerminalLeaves(){
    return root.getTerminalLeaves();
  }

  @Override
  public double predict(List<Double> features) {
    return root.calculateScore(features);
  }

  @Override
  public void writeModel(Ensemble.TreeConfig config, Writer writer) throws IOException {
    SavedModel savedModel = new SavedModel( getModelInfo(IntProp.FEATURE), getModelInfo(IntProp.ID), getModelInfo(DoubleProp.THRESHOLD), getModelInfo(DoubleProp.SCORE));
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, savedModel);
  }

  protected SavedModel getSavedModel(){ //"Suppress" config by making null.
    return new SavedModel( getModelInfo(IntProp.FEATURE), getModelInfo(IntProp.ID), getModelInfo(DoubleProp.THRESHOLD), getModelInfo(DoubleProp.SCORE));
  }

  public static class Split { //Node for trees
    private Split source;
    private Split leftLeaf; //TODO: Use List?
    private Split rightLeaf;
    private double threshold;
    private double score;
    private int featureId;
    private final List<Document> scoredDocs;
    private final int leafId;

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

    protected void fill(List<Double> info, DoubleProp type) {
      assert (type != null);
      double prop;
      if (type == DoubleProp.THRESHOLD)
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

    protected void fill(List<Integer> info, IntProp type) {
      assert(type != null);
      int prop;
      if (type == IntProp.FEATURE)
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

    public void setScore(double score) {
      this.score = score;
    }

    public boolean isRoot() {
      return source == null;
    }

    public Split getSource() {
      return source;
    }
  }

  protected enum IntProp {FEATURE, ID}
  protected enum DoubleProp {THRESHOLD, SCORE}

  protected static class SavedModel {

    public Ensemble.TreeConfig config;
    public List<Integer> leafIds;
    public List<Integer> featureIds;
    public List<Double> thresh;
    public List<Double> scores;

    SavedModel() {  // this is needed for Jackson...
    }

    SavedModel(Ensemble.TreeConfig config, List<Integer> featureIds, List<Integer> leafIds, List<Double> thresh, List<Double> scores) {
      this.config = config;
      this.featureIds = featureIds;
      this.leafIds = leafIds;
      this.thresh = thresh;
      this.scores = scores;
    }

    SavedModel(List<Integer> featureIds, List<Integer> leafIds, List<Double> thresh, List<Double> scores) {
      this.featureIds = featureIds;
      this.leafIds = leafIds;
      this.thresh = thresh;
      this.scores = scores;
    }
  }

}
