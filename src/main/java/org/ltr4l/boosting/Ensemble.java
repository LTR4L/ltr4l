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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;
import org.ltr4l.tools.Config;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

public class Ensemble extends Ranker<Ensemble.TreeConfig> {
  protected final List<RegressionTree> trees;

  public Ensemble(){
    trees = new ArrayList<>();
  }

  public Ensemble(Reader reader){
    trees = readModel(reader);
  }

  public void addTree(RegressionTree tree){
    trees.add(tree);
  }

  public List<RegressionTree> getTrees() {
    return trees;
  }

  public RegressionTree getTree(int i){
    return trees.get(i);
  }

  protected List<RegressionTree> readModel(Reader reader){
    throw new UnsupportedOperationException();
  }

  protected RegressionTree.SavedModel[] getTreeModels(){
    RegressionTree.SavedModel[] treeModels = new RegressionTree.SavedModel[trees.size()];
    for(int i = 0; i < trees.size(); i++)
      treeModels[i] = trees.get(i).getSavedModel();
    return treeModels;
  }

  @Override
  public void writeModel(Ensemble.TreeConfig config, Writer writer) throws IOException {
    SavedModel savedModel = new SavedModel(config, getTreeModels() );
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, savedModel);
  }

  @Override
  public double predict(List<Double> features) {
    return trees.stream().mapToDouble(tree -> tree.predict(features)).sum();
  }

  public static class TreeConfig extends Config {
    @JsonIgnore
    public int getNumTrees(){
      return getReqInt(params, "numTrees");
    }
    @JsonIgnore
    public int getNumLeaves(){
      return getReqInt(params, "numLeaves");
    }
    @JsonIgnore
    public double getLearningRate(){
      return getReqDouble(params, "learningRate");
    }
  }

  protected static class SavedModel {

    public TreeConfig config;
    public RegressionTree.SavedModel[] treeModels;

    SavedModel(){  // this is needed for Jackson...
    }

    SavedModel(TreeConfig config, RegressionTree.SavedModel[] treeModels){
      this.config = config;
      this.treeModels = treeModels;
    }
  }
}
