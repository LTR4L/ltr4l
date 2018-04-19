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

public class Tree {
  private Leaf sourceLeaf;
  private final List<Leaf> destinations; //index 0 = left, index 1 = right
  private double threshold;
  private int featureId;

  public Tree(int featureId, double threshold){
    sourceLeaf = null;
    this.featureId = featureId;
    destinations = new ArrayList<>();
    destinations.add(new Leaf(this)); //Left
    destinations.add(new Leaf(this)); //Right
    this.threshold = threshold;
  }

  public void setSourceLeaf(Leaf source) {
    this.sourceLeaf = source;
  }

  public Leaf getSourceLeaf() {
    return sourceLeaf;
  }

  public double score(List<Double> features){
    Leaf target = features.get(featureId) < threshold ? destinations.get(0) : destinations.get(1);
    return target.score(features);
  }

  public List<Leaf> getDestinationLeaves() {
    return destinations;
  }

  public Leaf getLeftLeaf(){
    return destinations.get(0);
  }

  public Leaf getRightLeaf(){
    return destinations.get(1);
  }

  public Leaf getDestinationLeaf(int i){
    return destinations.get(i);
  }

  public Tree getRootTree() {
    if (sourceLeaf == null) return this;
    return sourceLeaf.getSourceTree().getRootTree();
  }

  public boolean isRoot(){
    return sourceLeaf == null;
  }

  public List<Leaf> getOutputs(){
    return null; //TODO: IMPLEMENT
  }

  public static class Leaf { //Node
    private final Tree sourceTree;
    private Tree destination;
    private double score;
    private List<Document> results;

    public Leaf(Tree sourceTree){
      this.sourceTree = sourceTree;
      destination = null;
      results = new ArrayList<>();
    }


    public double score(List<Double> features) {
      if (destination == null){
        assert(results != null);
        return score;
      }
      else return destination.score(features);
    }

    public void setDestinationTree(Tree destination) {
      assert(destination == null || sourceTree.getRootTree() != destination.getRootTree());
      //assert(!isLinkedTo(destination) || destination == null); //TODO: Allow connection to non-linked part of tree?
      if(this.destination != null)  this.destination.setSourceLeaf(null); //Separate current destination reference to this leaf
      if(destination != null) destination.setSourceLeaf(this);
      this.destination = destination;
      results = null; //Documents can no longer land on this leaf.
    }

    public double getScore() {
      return score;
    }

    public void setScore(double score) {
      this.score = score;
    }

    public Tree getDestinationTree() {
      return destination;
    }

    public Tree getSourceTree() {
      return sourceTree;
    }

    protected boolean isLinkedTo(Tree tree){
      if(sourceTree == tree) return true;
      if(sourceTree.sourceLeaf == null) return false;
      return sourceTree.sourceLeaf.isLinkedTo(tree);
    }
  }
}

