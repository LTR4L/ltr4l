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

import com.fasterxml.jackson.databind.ObjectMapper;
import org.ltr4l.tools.Regularization;

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class NNMLP extends MLP {
  public NNMLP(int featureLength, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(featureLength, networkShape, optFact, regularization, weightModel);
  }

  public NNMLP(Reader reader) throws IOException {
    super(reader);
  }

  @Override
  protected void addOutputs(NetworkShape ns){
    return; //TODO: Implement addOutputs
  }

  @Override
  public double predict(List<Double> features) {
    double threshold = 0.5;
    forwardProp(features);
    for (int nodeId = 0; nodeId < network.get(network.size() - 1).size(); nodeId++) {
      MNode node = network.get(network.size() - 1).get(nodeId);
      if (node.getOutput() < threshold)
        return nodeId - 1;
    }
    return network.get(network.size() - 1).size() - 1;
  }

  @Override
  protected List<List<MNode>> readModel(Reader reader){
    try {
      ObjectMapper mapper = new ObjectMapper();
      SavedModel savedModel = mapper.readValue(reader, SavedModel.class);

      assert (savedModel.weights.size() > 0);

      List<List<MNode>> network = new ArrayList<>();
      List<MNode> currentLayer = new ArrayList<>();

      // in order to get dim, see the number of input edges of the first node in the first hidden layer
      // "-1" is bias
      final int inputDim = savedModel.getNode(0, 0).size() - 1;
      //Start with constructing the input layer.
      for (int i = 0; i < inputDim; i++) {
        currentLayer.add(constructNode(new Activation.Identity()));
      }
      network.add(currentLayer);

      //Construct hidden layers
      NetworkShape networkShape = savedModel.config.getNetworkShape();
      //addOutputs(networkShape); After addOutputs has been properly implemented, overriding readModel will be unnecessary.
      int outputNum = savedModel.getLayer(savedModel.weights.size() - 1).size();
      networkShape.add(outputNum, new Activation.Sigmoid());
      Optimizer.OptimizerFactory optFact = savedModel.config.getOptFact();
      for (int layerNum = 0; layerNum < savedModel.weights.size(); layerNum++) {
        currentLayer = new ArrayList<>();
        network.add(currentLayer);
        int nodeNum = savedModel.getLayer(layerNum).size();
        Activation activation = networkShape.getLayerSetting(layerNum).getActivation();

        for (int i = 0; i < nodeNum; i++) {
          MNode currentNode = constructNode(activation);
          Optimizer opt = optFact.getOptimizer();
          int k = 0;
          currentNode.addInputEdge(constructEdge(null, currentNode, opt, savedModel.getWeight(layerNum, i, k++))); //add bias edge
          currentLayer.add(currentNode);

          for (MNode previousNode : network.get(layerNum)) {    //Note network.get(layerNum-1) is previous layer!
            Edge edge = constructEdge(previousNode, currentNode, optFact.getOptimizer(), savedModel.getWeight(layerNum, i, k++));
            currentNode.addInputEdge(edge);
            previousNode.addOutputEdge(edge);
          }
        }
      }
      return network;
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }
}
