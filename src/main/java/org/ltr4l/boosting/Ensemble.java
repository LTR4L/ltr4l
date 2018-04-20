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
import org.ltr4l.Ranker;
import org.ltr4l.tools.Config;

import java.io.IOException;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;

public class Ensemble extends Ranker<Ensemble.TreeConfig> {
  protected final List<RegressionTree> trees;

  public Ensemble(){
    trees = new ArrayList<>();
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

  @Override
  public void writeModel(Ensemble.TreeConfig config, Writer writer) throws IOException {
    throw new UnsupportedOperationException(); //TODO: Implement
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
}
