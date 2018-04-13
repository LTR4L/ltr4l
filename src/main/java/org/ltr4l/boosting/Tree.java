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

import java.util.ArrayList;
import java.util.List;

public class Tree {
  private Leaf sourceLeaf;
  private final List<Leaf> destinations;
  private double threshold;
  private int featureId;
  private static double DEFAULT_THRESHOLD = 5;


  public Tree(int featureId){
    sourceLeaf = null;
    this.featureId = featureId;
    destinations = new ArrayList<>();
    Leaf leaf1 = new Leaf(this, 0);
    Leaf leaf2 = new Leaf(this, 0);
    destinations.add(leaf1);
    destinations.add(leaf2);
    threshold = DEFAULT_THRESHOLD;
  }

  public Tree(int featureId, double threshold, double... destinationScores){
    assert(destinationScores.length == 2); //TODO: For now, limit to 2.
    sourceLeaf = null;
    this.featureId = featureId;
    destinations = new ArrayList<>();

    for(int i = 0; i < destinationScores.length; i++)
      destinations.add(new Leaf(this, destinationScores[i]));
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

  public Leaf getDestinationLeaf(int i){
    return destinations.get(i);
  }

  public Tree getRootTree() {
    if (sourceLeaf == null) return this;
    return sourceLeaf.getSourceTree().getRootTree();
  }

  public static class Leaf { //Node
    private final Tree sourceTree;
    private Tree destination;
    private double score;

    public Leaf(Tree sourceTree, double score){
      this.sourceTree = sourceTree;
      this.score = score;
      destination = null;
    }


    public double score(List<Double> features) {
      if (destination == null) return score;
      else return destination.score(features);
    }

    public void setDestinationTree(Tree destination) {
      assert(destination == null || sourceTree.getRootTree() != destination.getRootTree());
      //assert(!isLinkedTo(destination) || destination == null); //TODO: Allow connection to non-linked part of tree?
      if(this.destination != null)  this.destination.setSourceLeaf(null); //Separate current destination reference to this leaf
      if(destination != null) destination.setSourceLeaf(this);
      this.destination = destination;
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

