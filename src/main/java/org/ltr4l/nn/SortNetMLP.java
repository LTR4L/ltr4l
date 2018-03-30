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

import java.util.ArrayList;
import java.util.List;

import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;
import org.ltr4l.trainers.MLPTrainer;

public class SortNetMLP extends AbstractMLPBase<SortNetMLP.SNode, SortNetMLP.SEdge> {

  //Construct Network
  public SortNetMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(inputDim, networkShape, optFact, regularization, weightModel);
  }

  public SortNetMLP(int inputDim, MLPTrainer.MLPConfig config){
    super(inputDim, config);
  }

  @Override
  protected List<List<SNode>> constructNetwork(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact){
    List<List<SNode>> network = new ArrayList<>();

    //Construct the initial layer:
    List<SNode> inputLayer = new ArrayList<>();
    List<SNode> inputLayerPrime = new ArrayList<>();
    network.add(inputLayer);
    for (int i = 0; i < inputDim; i++) {
      inputLayer.add(new SNode(0, new Activation.Identity()));
      inputLayerPrime.add(new SNode(1, new Activation.Identity()));
    }
    inputLayer.addAll(inputLayerPrime);

    //Construct the rest of the layers and edges:
    final double bias = weightInit.getInitialBias();
    for (int layerId = 0; layerId < networkShape.size(); layerId++) {
      int numNodes = networkShape.getLayerSetting(layerId).getNum();
      Activation activation = networkShape.getLayerSetting(layerId).getActivation();

      List<SNode> currentLayer = new ArrayList<>();
      List<SNode> layerPrime = new ArrayList<>();
      network.add(currentLayer);

      for (int i = 0; i < numNodes; i++) {
        SNode sNode0 = new SNode(0, activation);
        SNode sNode1 = new SNode(1, activation);
        currentLayer.add(sNode0);
        layerPrime.add(sNode1);
        SNode[] sNodePair = {sNode0, sNode1};

        //Add bias
        SEdge biasEdge = new SEdge(null, sNodePair, optFact.getOptimizer(), bias);
        sNode0.addInputEdge(biasEdge);
        sNode1.addInputEdge(biasEdge);

        //Add edges with previous layer.
        List<SNode> prevLayer = network.get(layerId);
        for (int nodeId = 0; nodeId < prevLayer.size() / 2; nodeId++) {
          SNode prevSNode0 = prevLayer.get(nodeId);
          SNode prevSNode1 = prevLayer.get(nodeId + prevLayer.size() / 2);
          double weight = weightInit.getNextRandomInitialWeight();
          SNode[] prevNodePair = {prevSNode0, prevSNode1};
          SEdge sEdge = new SEdge(prevNodePair, sNodePair, optFact.getOptimizer(), weight);

          prevSNode0.addOutputEdge(sEdge);
          prevSNode1.addOutputEdge(sEdge);
          sNode0.addInputEdge(sEdge);
          sNode1.addInputEdge(sEdge);

          //Get another weight, and set up an edge for reversed pair.
          weight = weightInit.getNextRandomInitialWeight();
          prevNodePair = new SNode[]{prevSNode1, prevSNode0};
          sEdge = new SEdge(prevNodePair, sNodePair, optFact.getOptimizer(), weight);
          prevSNode1.addOutputEdge(sEdge);
          prevSNode0.addOutputEdge(sEdge);
          sNode0.addInputEdge(sEdge);
          sNode1.addInputEdge(sEdge);
        }
      }
      currentLayer.addAll(layerPrime);
    }
    return network;
  }

  @Override
  protected void addOutputs(NetworkShape networkShape){
    networkShape.add(1, new Activation.Sigmoid());
  }

  public double[] getOutputs(){
    int layer = network.size() - 1;
    return new double[] {getNode(layer, 0).getOutput(), getNode(layer, 1).getOutput()};
  }

  // if > 0, doc1 is predicted to be more relevant than doc2
  // if < 0, doc1 is predicted to be less relevant than doc 2.
  public double predict(Document doc1, Document doc2) {
    List<Double> combinedFeatures = new ArrayList<>(doc1.getFeatures());
    combinedFeatures.addAll(doc2.getFeatures());
    return predict(combinedFeatures);
  }

  @Override
  public double predict(List<Double> features){
    forwardProp(features);
    double[] output = getOutputs();
    return output[0] - output[1];
  }

  public double forwardProp(List<Double> features) {
    //First, feed features into input layer:
    for (int i = 0; i < network.get(0).size(); i++) {
      SNode node = network.get(0).get(i);
      double feature = features.get(i);
      node.setOutput(feature);
    }

    //Then forward propagate:
    for (int layerId = 1; layerId < network.size(); layerId++) {
      List<SNode> layer = network.get(layerId);
      for (SNode node : layer) {
        node.updateOutput();
      }
    }

    List<SNode> outputLayer = network.get(network.size() - 1);
    return outputLayer.get(0).getOutput();
  }

  //This only sets the first node.
  //Use backProp(double target, Error errorFunc)!!
  @Override
  public void backProp(Error errorFunc, double... target) {
    setOutputLayerDerivatives(errorFunc, target);
    for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) { //When going through each layer, you modify the previous layer.
      List<SNode> layer = network.get(layerIdx);

      for (SNode node : layer) {
        // Second, find ∂C/∂I by (∂C/∂O)(∂O/∂I)
        // I = total Input; O = output = Activation(I)
        double totalInput = node.getTotalInput();
        double inDer = node.getActivation().derivative(totalInput) * node.getOutputDer();
        node.setInputDer(inDer);

        //First edge is bias.
        SEdge edge = node.getInputEdges().get(0);

        double accErrDer = edge.getAccErrorDer();
        accErrDer += node.getInputDer();
        edge.setAccErrorDer(accErrDer);

        for (int edgeNum = 1; edgeNum < node.getInputEdges().size(); edgeNum++) {
          edge = node.getInputEdges().get(edgeNum);
          if (!edge.isDead()) {
            //(∂C/∂I)*(∂I/∂w) = Σ∂C/∂Ii *(∂Ii/∂w) = ∂C/∂w
            //(∂Ii/∂w) = Oi, because Oi*wi = Ii
            double errorDer = node.getInputDer() * edge.getSource()[node.getGroup()].getOutput(); //(∂C/∂I)*(∂I/∂w)
            accErrDer = edge.getAccErrorDer();
            accErrDer += errorDer;
            edge.setAccErrorDer(accErrDer);
          }
        }
      }
      if (layerIdx != 1) {
        List<SNode> previousLayer = network.get(layerIdx - 1);
        for (SNode node : previousLayer) {
          double oder = 0;
          for (SEdge outEdge : node.getOutputEdges()) {
            //∂C/∂Oi = ∂Ik/∂Oi * ∂C/∂Ik
            oder += outEdge.getWeight() * outEdge.getDestination()[node.getGroup()].getInputDer();
          }
          node.setOutputDer(oder);
        }
      }
    }
    numAccumulatedDer += 1;
  }

  //Same implementation as MLP class...
  @Override
  public void updateWeights(double lrRate, double rgRate) {
    //Update all weights in all edges.
    for (int layerId = 1; layerId < network.size(); layerId++) {  //All Layers
      List<SNode> layer = network.get(layerId);
      for (SNode node : layer) {                     //All Nodes
        for (SEdge edge : node.getInputEdges()) { //All edges for each node.
          if (!edge.isDead()) {
            double rgDer = 0;
            if (regularization != null)
              rgDer = regularization.derivative(edge.getWeight());

            if (numAccumulatedDer > 0) {
              //"Optimize" dw; ηdw
              double cost = edge.getOptimizer().optimize(edge.getAccErrorDer(), lrRate, iter);
              double weight = edge.getWeight();
              weight += cost / numAccumulatedDer; //accumulated many dw, so avg that.

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
    }
    numAccumulatedDer = 0; //Weights updated, now no derivatives have been accumulated.
    iter++;
  }

  /**
   * Similar to Edge, however each source and destination contains two nodes.
   * This is to preserve symmetry in the weights in the network.
   * For example, LayerA = {node1, node2, ..., node1', node2'}, LayerB = {nodeA, nodeB, ..., nodeA', nodeB'}.
   * node1:nodeA = node1':nodeA' and node1:nodeA' = node1':nodeA,
   * where nodeX:nodeY = weight between node x and node y.
   */
  protected static class SEdge extends AbstractEdge.AbstractCmpEdge<SNode> {
    private double accErrorDer;

    protected SEdge(SNode[] source, SNode[] destination, Optimizer optimizer, double weight) {
      super(source, destination, optimizer, weight);
      accErrorDer = 0.0;
    }

    public void setAccErrorDer(double accErrorDer) {
      this.accErrorDer = accErrorDer;
    }

    public double getAccErrorDer() {
      return accErrorDer;
    }
  }

  protected static class SNode extends AbstractNode<SEdge> {
    private final int group;

    protected SNode(int group, Activation activation) {
      super(activation);
      this.group = group;
    }

    @Override
    public void updateOutput() {
      totalInput = inputEdges.get(0).getWeight();
      for (int i = 1; i < inputEdges.size(); i++) {
        totalInput += inputEdges.get(i).getWeight() * inputEdges.get(i).getSource()[group].getOutput();
      }
      output = activation.output(totalInput);
    }

    public int getGroup() {
      return group;
    }
  }
}
