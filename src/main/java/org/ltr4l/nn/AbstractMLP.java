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

import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.ltr4l.tools.Regularization;

/**
 * This is the AbstractMLP class for standard Feed-Forward Neural Networks.
 *
 */
public abstract class AbstractMLP <N extends AbstractNode.Node, E extends AbstractEdge.AbstractFFEdge> extends AbstractMLPBase<N, E> {

  public AbstractMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(inputDim, networkShape, optFact, regularization, weightModel);
  }

  protected List<List<N>> constructNetwork(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact){
    List<List<N>> network = new ArrayList<>();
    List<N> currentLayer = new ArrayList<>();

    //Start with constructing the input layer.
    for (int i = 0; i < inputDim; i++) {
      currentLayer.add(constructNode(new Activation.Identity()));
    }
    network.add(currentLayer);

    //Construct hidden layers
    final double bias = weightInit.getInitialBias();
    for (int layerNum = 0; layerNum < networkShape.size(); layerNum++) {
      currentLayer = new ArrayList<>();
      network.add(currentLayer);
      int nodeNum = networkShape.getLayerSetting(layerNum).getNum();
      Activation activation = networkShape.getLayerSetting(layerNum).getActivation();

      for (int i = 0; i < nodeNum; i++) {
        N currentNode = constructNode(activation);
        Optimizer opt = optFact.getOptimizer();
        currentNode.addInputEdge(constructEdge(null, currentNode, opt, bias)); //add bias edge
        currentLayer.add(currentNode);

        for (N previousNode : network.get(layerNum)) {    //Note network.get(layerNum) is previous layer!
          E edge = constructEdge(previousNode, currentNode, optFact.getOptimizer(), weightInit.getNextRandomInitialWeight());
          currentNode.addInputEdge(edge);
          previousNode.addOutputEdge(edge);
        }
      }
    }
    return network;
  }

  protected abstract N constructNode(Activation activation);
  protected abstract E constructEdge(N source, N destination, Optimizer opt, double weight);

  static class ModelReader <N extends AbstractNode.Node, E extends AbstractEdge.AbstractFFEdge> {

    // TODO: use Factory...?
    // don't want to have dummy, it is needed to call constructNode & constructEdge
    public List<List<N>> readModel(Reader reader, AbstractMLP<N, E> dummy) throws IOException {
      ObjectMapper mapper = new ObjectMapper();
      SavedModel savedModel = mapper.readValue(reader, SavedModel.class);

      assert(savedModel.weights.size() > 0);

      List<List<N>> network = new ArrayList<>();
      List<N> currentLayer = new ArrayList<>();

      // in order to get dim, see the number of input edges of the first node in the first hidden layer
      // "-1" is bias
      final int inputDim = savedModel.getNode(0, 0).size() - 1;
      //Start with constructing the input layer.
      for (int i = 0; i < inputDim; i++) {
        currentLayer.add(dummy.constructNode(new Activation.Identity()));
      }
      network.add(currentLayer);

      //Construct hidden layers
      NetworkShape networkShape = savedModel.config.getNetworkShape();
      Optimizer.OptimizerFactory optFact = savedModel.config.getOptFact();
      for (int layerNum = 0; layerNum < savedModel.weights.size(); layerNum++) {
        currentLayer = new ArrayList<>();
        network.add(currentLayer);
        int nodeNum = savedModel.getLayer(layerNum).size();
        Activation activation = networkShape.getLayerSetting(layerNum).getActivation();

        for (int i = 0; i < nodeNum; i++) {
          N currentNode = dummy.constructNode(activation);
          Optimizer opt = optFact.getOptimizer();
          int k = 0;
          currentNode.addInputEdge(dummy.constructEdge(null, currentNode, opt, savedModel.getWeight(layerNum, i, k++))); //add bias edge
          currentLayer.add(currentNode);

          for (N previousNode : network.get(layerNum)) {    //Note network.get(layerNum-1) is previous layer!
            E edge = dummy.constructEdge(previousNode, currentNode, optFact.getOptimizer(), savedModel.getWeight(layerNum, i, k++));
            currentNode.addInputEdge(edge);
            previousNode.addOutputEdge(edge);
          }
        }
      }
      return network;
    }
  }
}
