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

import org.ltr4l.tools.Regularization;

public class RankNetMLP extends MLP {

  public RankNetMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(inputDim, networkShape, optFact, regularization, weightModel);
  }

  @Override
  protected void addOutputs(NetworkShape ns) {
    ns.add(1, new Activation.Identity());
  }

  /**
   * Backpropagation is very similar to MLP, however the derivative used in backpropagation is calculated first, and then
   * passed to the method. This is so that all RankNet based classes can use the same implementation for backpropagation.
   * @param lambda
   */
  public void backProp(double lambda) {
    Node outputNode = network.get(network.size() - 1).get(0);

    //First, get the derivative ∂C/∂O and set it to output derivative of the final node.
    outputNode.setOutputDer(lambda);

    //When going through each layer, modify the previous layer.
    for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) {
      List<Node> layer = network.get(layerIdx);

      for (Node node : layer) {
        // Second, find ∂C/∂I by (∂C/∂O)(∂O/∂I)
        // I = total Input; O = output = Activation(I)
        double totalInput = node.getTotalInput();
        double inDer = node.getActivation().derivative(totalInput) * node.getOutputDer();
        node.setInputDer(inDer);

        //First edge is bias.
        Edge edge = node.getInputEdges().get(0);

        double accErrDer = edge.getAccErrorDer();
        accErrDer += node.getInputDer();
        edge.setAccErrorDer(accErrDer);

        for (int edgeNum = 1; edgeNum < node.getInputEdges().size(); edgeNum++) {
          edge = node.getInputEdges().get(edgeNum);
          if (!edge.isDead()) {
            //(∂C/∂I)*(∂I/∂w) = Σ∂C/∂Ii *(∂Ii/∂w) = ∂C/∂w
            //(∂Ii/∂w) = Oi, because Oi*wi = Ii
            double errorDer = node.getInputDer() * edge.getSource().getOutput(); //(∂C/∂I)*(∂I/∂w)
            accErrDer = edge.getAccErrorDer();
            accErrDer += errorDer;
            edge.setAccErrorDer(accErrDer);
          }
        }
      }
      if (layerIdx != 1) {
        List<Node> previousLayer = network.get(layerIdx - 1);
        for (Node node : previousLayer) {
          double oder = 0;
          for (Edge outEdge : node.getOutputEdges()) {
            //∂C/∂Oi = ∂Ik/∂Oi * ∂C/∂Ik
            oder += outEdge.getWeight() * outEdge.getDestination().getInputDer();
          }
          node.setOutputDer(oder);
        }
      }
    }
    //numAccumulatedDer += 1;
  }

  @Override
  public void updateWeights(double lrRate, double rgRate) {
    //Update all weights in all edges.
    for (int layerId = 1; layerId < network.size(); layerId++) {  //All Layers
      List<Node> layer = network.get(layerId);
      for (Node node : layer) {                     //All Nodes
        for (Edge edge : node.getInputEdges()) { //All edges for each node.
          if (!edge.isDead()) {
            double rgDer = 0;
            if (regularization != null)
              rgDer = regularization.derivative(edge.getWeight());

            //"Optimize" dw; ηdw
            double cost = edge.getOptimizer().optimize(edge.getAccErrorDer(), lrRate, iter);
            double weight = edge.getWeight();
            weight += cost; //Only accumulated dw over one pass.

            //Wi = Wi + (learning rate/optimization) * ∂C/∂w
            //Further update weight based on regularization.
            double newWeight = weight - rgDer * (lrRate * rgRate);
            if (regularization instanceof Regularization.L1 && weight * newWeight < 0) {
              edge.setWeight(0d);
              edge.setDead(true);
            } else {
              edge.setWeight(newWeight);
            }
            //Wi = Wi + (learning rate/optimization) * ∂C/∂w
            edge.setAccErrorDer(0); //After the change has been applied, reset the accumulated derivatives.

          }
        }
      }
    }
    //Weights updated, derivatives will not be accumulated.
    iter++;
  }
}
