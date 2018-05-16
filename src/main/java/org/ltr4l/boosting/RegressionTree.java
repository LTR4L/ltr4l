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

  public RegressionTree(int numLeaves, int initFeat, double initThreshold, List<Document> docs) throws InvalidFeatureThresholdException{
    assert(numLeaves >= 2);
    root = new Split(initFeat, initThreshold, docs);
    Map<Split, OptimalLeafLoss> splitErrorMap = new HashMap<>();
    for(int l = 2; l < numLeaves; l++) {
      for (Split leaf : root.getTerminalLeaves()) {
        if(!splitErrorMap.containsKey(leaf)) //Speedup: only calculate if it hasnt been done so yet... should be twice
          splitErrorMap.put(leaf, TreeTools.findMinLeafThreshold(leaf.getScoredDocs()));
      }
      Split optimalLeaf = TreeTools.findOptimalLeaf(splitErrorMap);
      int feature = splitErrorMap.get(optimalLeaf).getOptimalFeature();
      double threshold = splitErrorMap.get(optimalLeaf).getOptimalThreshold();
      optimalLeaf.addSplit(feature, threshold);
      splitErrorMap.remove(optimalLeaf);
    }
  }

  public RegressionTree(SavedModel model){
    int numNodes = model.leafIds.size();
    assert(numNodes > 3);
    root = new Split(null, model.featureIds.get(0), model.thresh.get(0), model.leafIds.get(0), model.scores.get(0));
    Split currentNode = root;
    for(int i = 1; i < numNodes; i++){
      int currentId = currentNode.getLeafId();
      int featId = model.featureIds.get(i);
      double nextThresh = model.thresh.get(i);
      int nextId = model.leafIds.get(i);
      double nextScore = model.scores.get(i);

      if(nextId == 2 * currentId + 1){
        currentNode.setLeftLeaf(new Split(currentNode, featId, nextThresh, nextId, nextScore));
        currentNode = currentNode.getLeftLeaf();
      }

      else if(nextId == 2 * currentId + 2){
        currentNode.setRightLeaf(new Split(currentNode, featId, nextThresh, nextId, nextScore));
        currentNode = currentNode.getRightLeaf();
      }

      else{
        currentNode = currentNode.getSource(); //Go back up the tree
        i--;
      }
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

  protected Split getRoot() {
    return root;
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
