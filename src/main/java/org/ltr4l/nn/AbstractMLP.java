/*
 * Copyright 2018 org.LTR4L
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ltr4l.nn;

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Properties;
import java.util.stream.Collectors;

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

}
