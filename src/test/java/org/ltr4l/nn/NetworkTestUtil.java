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

package org.ltr4l.nn;

import java.util.List;

import org.junit.Assert;

public class NetworkTestUtil<N extends AbstractNode> {

  public void assertLayer(List<List<N>> model, int layerIdx, String expected) throws Exception {
    List<N> layer = model.get(layerIdx);
    String[] nodes = expected.split("\\|", -1);
    Assert.assertEquals(nodes.length, layer.size());
    for(int i = 0; i < nodes.length; i++){
      assertNodeInputEdges(layer, i, nodes[i]);
    }
  }

  static final double WEIGHT_ERROR = 0.001;

  public void assertNodeInputEdges(List<N> layerModel, int nodeIdx, String expected) throws Exception {
    N node = layerModel.get(nodeIdx);
    if(expected.isEmpty()){
      Assert.assertEquals(0, node.getInputEdges().size());
    }
    else{
      String[] edges = expected.split(",", -1);
      Assert.assertEquals(edges.length, node.getInputEdges().size());
      for(int i = 0; i < edges.length; i++){
        Assert.assertEquals(Double.parseDouble(edges[i]), node.getInputEdge(i).getWeight(), WEIGHT_ERROR);
      }
    }
  }

  public void assertCmpLayer(List<List<N>> model, int layerIdx, String expected) throws Exception {
    List<N> layer = model.get(layerIdx);
    int prevLayerSize = layerIdx <= 0 ? 0 : model.get(layerIdx - 1).size();
    String[] nodes = expected.split("\\|", -1);
    Assert.assertEquals(nodes.length, layer.size());
    for(int i = 0; i < nodes.length; i++){
      assertCmpNodeInputEdges(layer, i, nodes[i], prevLayerSize);
    }
  }

  public void assertCmpNodeInputEdges(List<N> layerModel, int nodeIdx, String expected, int prevLayerSize) throws Exception {
    N node = layerModel.get(nodeIdx);
    if (expected.isEmpty()){
      Assert.assertEquals(0, node.getInputEdges().size());
    }
    else{
      String[] edges = expected.split(",", -1);
      Assert.assertEquals(edges.length, node.getInputEdges().size() - prevLayerSize/2);
    }
  }
}
