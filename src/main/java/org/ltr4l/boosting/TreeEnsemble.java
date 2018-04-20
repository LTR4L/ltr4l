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
import org.ltr4l.tools.Config;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class TreeEnsemble extends Ranker<TreeEnsemble.TreeConfig> {
  private final List<Tree> trees;

  public TreeEnsemble(List<Tree> initialTrees){
    trees = initialTrees;
  }

  public TreeEnsemble(){
    trees = new ArrayList<>();
  }

  public void addTree(Tree tree){
    assert(tree.isRoot());
    trees.add(tree);
  }

  @Override
  public void writeModel(TreeConfig config, Writer writer) throws IOException{
    return; //TODO: Implement
  }

  @Override
  public double predict(List<Double> features) { //TODO: Add weights to trees?
    return trees.isEmpty() ? 0 : trees.stream().mapToDouble(tree -> tree.score(features)).sum();
  }

  public static class TreeConfig extends Config {
    public int getNumTrees(){
      return getReqInt(params, "numTrees");
    }
    public int getNumLeaves(){
      return getReqInt(params, "numLeaves");
    }
  }

}
