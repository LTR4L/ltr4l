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

  protected double[] getDoubleInfo(FillType type){
    double[] info = new double[2 * numLeaves]; //For now, get for all leaves. Total is 2n - 1, but skipping index 0.
    info[0] = Double.NaN; //Don't use 0th index, to make life easier.
    switch(type){
      case THRESHOLD:
        root.fillThresholds(info);
        break;
      case SCORE:
        root.fillScores(info);
        break;
      default:
        throw new IllegalArgumentException();
    }
    return Arrays.copyOfRange(info, 1, info.length);
  }

  protected int[] getIntInfo(FillType type){
    int[] info = new int[2 * numLeaves];
    info[0] = -2; //-2 should not occur during training. TODO: List or Integer
    switch(type){
      case FEATURE:
        root.fillFeatures(info);
        break;
      case ID:
        root.fillIds(info);
        break;
      default:
        throw new IllegalArgumentException();
    }
    return Arrays.copyOfRange(info, 1, info.length);
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
    SavedModel savedModel = new SavedModel( getIntInfo(FillType.FEATURE), getIntInfo(FillType.ID), getDoubleInfo(FillType.THRESHOLD), getDoubleInfo(FillType.SCORE));
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, savedModel);
  }

  protected SavedModel getSavedModel(){ //"Suppress" config by making null.
    return new SavedModel( getIntInfo(FillType.FEATURE), getIntInfo(FillType.ID), getDoubleInfo(FillType.THRESHOLD), getDoubleInfo(FillType.SCORE));
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

    protected Split(int featureId, double threshold, List<Document> scoredDocs) throws InvalidFeatureThresholdException{ //For root node.
      this.source = null;
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
      leftLeaf = new Split(this, leftDocs, 2);
      rightLeaf = new Split(this, rightDocs, 3);
      leafId = 1;
    }

    protected Split(Split source, List<Document> scoredDocs, int leafId) throws InvalidFeatureThresholdException{
      if(scoredDocs.isEmpty()) throw new InvalidFeatureThresholdException();
      this.source = source;
      this.scoredDocs = scoredDocs;
      this.leafId = leafId;
      leftLeaf = null;
      rightLeaf = null;
      score = 0.0d;
      threshold = Double.NEGATIVE_INFINITY;
      featureId = -1;
    }

    protected void addSplit(int feature, double threshold) throws InvalidFeatureThresholdException{
      this.featureId = feature;
      this.threshold = threshold;
      List<Document> leftDocs = new ArrayList<>();
      List<Document> rightDocs = new ArrayList<>();
      for(Document doc : this.scoredDocs){
        if(doc.getFeature(featureId) < threshold) leftDocs.add(doc);
        else rightDocs.add(doc);
      }
      leftLeaf = new Split(this, leftDocs, 2 * leafId);
      rightLeaf = new Split(this, rightDocs, (2 * leafId) + 1);
    }

    protected List<Split> getTerminalLeaves(){
      List<Split> terminalLeaves = new ArrayList<>();
      if (!hasDestinations()){
        terminalLeaves.add(this);
        return terminalLeaves;
      }
      getDestinations().forEach(leaf -> terminalLeaves.addAll(leaf.getTerminalLeaves()));
      return terminalLeaves;
    }

    protected double calculateScore(List<Double> features){
      assert(leavesProperlySet());
      if(!hasDestinations()) return score;
      Split destination = features.get(featureId) < threshold ? leftLeaf : rightLeaf;
      return destination.calculateScore(features);
    }

    protected void fillScores(double[] scores){
      int index;
      if(leafId == 1 || leafId == 2 || leafId == 3) index = leafId;
      else index = leafId % 2 == 1 ? (leafId - 1) / 2 : leafId / 2;
      scores[index] = score;
      if(hasDestinations()) getDestinations().forEach(leaf -> leaf.fillScores(scores));
    }

    protected void fillThresholds(double[] thresholds){
      int index;
      if(leafId == 1 || leafId == 2 || leafId == 3) index = leafId;
      else index = leafId % 2 == 1 ? (leafId - 1) / 2 : leafId / 2;
      thresholds[index] = score;
      if(hasDestinations()) getDestinations().forEach(leaf -> leaf.fillThresholds(thresholds));
    }

    protected void fillFeatures(int[] leafFeatures){
      int index;
      if(leafId == 1 || leafId == 2 || leafId == 3) index = leafId;
      else index = leafId % 2 == 1 ? (leafId - 1) / 2 : leafId / 2;
      leafFeatures[index] = featureId;
      if(hasDestinations()) getDestinations().forEach(leaf -> leaf.fillFeatures(leafFeatures));
    }

    protected void fillIds(int[] ids){
      int index;
      if(leafId == 1 || leafId == 2 || leafId == 3) index = leafId;
      else index = leafId % 2 == 1 ? (leafId - 1) / 2 : leafId / 2;
      ids[index] = leafId;
      if(hasDestinations()) getDestinations().forEach(leaf -> leaf.fillIds(ids));
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

  protected enum FillType{THRESHOLD, SCORE, FEATURE, ID}

  protected static class SavedModel {

    public Ensemble.TreeConfig config;
    public int[] featureIds;
    public int[] leafIds;
    public double[] thresh;
    public double[] scores;

    SavedModel(){  // this is needed for Jackson...
    }

    SavedModel(Ensemble.TreeConfig config, int[] featureIds, int[] leafIds, double[] thresh, double[] scores){
      this.config = config;
      this.featureIds = featureIds;
      this.leafIds = leafIds;
      this.thresh = thresh;
      this.scores = scores;
    }

    SavedModel(int[] featureIds, int[] leafIds, double[] thresh, double[] scores){
      this.featureIds = featureIds;
      this.leafIds = leafIds;
      this.thresh = thresh;
      this.scores = scores;
    }
  }

}
