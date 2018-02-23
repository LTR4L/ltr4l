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
import java.util.Random;
import java.util.stream.Collectors;

import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;

public class MLP {
  protected List<List<Node>> network;
  protected long iter;
  protected int numAccumulatedDer;
  protected int nWeights;
  protected Regularization regularization;

  //CONSTRUCT NETWORK
  public MLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    //Network shape describes number of nodes and their activation. Example:
    //[
    //[12, Sigmoid ]
    //[4 , Softmax ]
    //[1 , Identity]
    //]
    iter = 1;
    numAccumulatedDer = 0;
    this.regularization = regularization;
    nWeights = inputDim * networkShape.getLayerSetting(0).getNum();  //Number of weights used for Xavier initialization.
    network = new ArrayList<>();

    for (int i = 1; i < networkShape.size(); i++) {
      nWeights += networkShape.getLayerSetting(i - 1).getNum() * networkShape.getLayerSetting(i).getNum();
    }

    //Start with constructing the input layer
    List<Node> currentLayer = new ArrayList<>();
    for (int i = 0; i < inputDim; i++) {
      currentLayer.add(new Node(new Activation.Identity()));
    }
    network.add(currentLayer);

    //Construct hidden layers
    for (int layerNum = 0; layerNum < networkShape.size(); layerNum++) {
      currentLayer = new ArrayList<>();
      network.add(currentLayer);
      int nodeNum = networkShape.getLayerSetting(layerNum).getNum();
      Activation activation = networkShape.getLayerSetting(layerNum).getActivation();
      double bias = weightModel.toLowerCase().equals("zero") ? 0 : 0.01;

      for (int i = 0; i < nodeNum; i++) {
        Node currentNode = new Node(activation);
        Optimizer opt = optFact.getOptimizer();
        currentNode.addInputEdge(new Edge(null, currentNode, opt, bias)); //add bias edge
        currentLayer.add(currentNode);

        for (Node previousNode : network.get(layerNum)) {    //Note network.get(layerNum) is previous layer!
          Edge edge = new Edge(previousNode, currentNode, optFact.getOptimizer(), weightInit(weightModel));
          currentNode.addInputEdge(edge);
          previousNode.addOutputEdge(edge);
        }
      }
    }
  }

  private double weightInit(String init) {
    init.toLowerCase();
    switch (init) {
      case "xavier":
        return new Random().nextGaussian() / nWeights;
      case "normal":
        return new Random().nextGaussian();
      case "uniform":
        return new Random().nextDouble();
      case "zero":
        return 0d;
      default:
        return new Random().nextGaussian();
    }
  }

  public List<List<List<Double>>> getWeights() {
    //Note: went with collect as it is necessary to get a list of all weights anyway.
    return network.stream().map(layer -> layer.stream()
        .map(node -> node.getOutputEdges().stream()
            .map(Edge::getWeight)
            .collect(Collectors.toList()))
        .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  public double predict(Document doc) {
    return forwardProp(doc);
  }

  //Feed forward propagation
  //This is for the case of the output layer with one node.
  public double forwardProp(List<Double> features) {
    //Feed document features into input layer.
    List<Node> layer = network.get(0); //First layer
    for (int i = 0; i < layer.size(); i++) {
      Node node = layer.get(i);
      node.setOutput(features.get(i));
    }

    //Go through the rest of the layers and update the output.
    for (int layerId = 1; layerId < network.size(); layerId++) {
      layer = network.get(layerId);
      for (Node node : layer) {
        node.updateOutput();
      }
    }
    return layer.get(0).getOutput(); //After the loop, layer = lastLayer
  }

  //Overloaded to accept Document object as well.
  public double forwardProp(Document doc) {
    List<Double> features = doc.getFeatures();
    return forwardProp(features);

  }

  //This is for one output node.
  public void backProp(double target, Error errorFunc) {
    Node outputNode = network.get(network.size() - 1).get(0);
    double output = outputNode.getOutput();
    //First, get the derivative ∂C/∂O and set it to output derivative of the final node.
    double der = errorFunc.der(output, target);
    outputNode.setOutputDer(der);

    for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) { //When going through each layer, you modify the previous layer.
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
          //double oder = node.getOutputDer();
          //node.setOutputDer(0);
          node.setOutputDer(0);
          double oder = 0;
          for (Edge outEdge : node.getOutputEdges()) {
            //And finally, ∂C/∂w = ∂C/∂I * ∂I/∂w
            oder += outEdge.getWeight() * outEdge.getDestination().getInputDer();
            node.setOutputDer(oder);
          }
        }
      }
    }
    numAccumulatedDer += 1;
  }

  //This is for the case of multiple output layers.
  public void backProp(double[] targets, Error errorFunc) {
    //First, feed derivative into each node in output layer
    //Skip the first node, as the derivative will be set through backprop method.
    List<Node> outputLayer = network.get(network.size() - 1);
    for (int i = 1; i < outputLayer.size(); i++) {
      Node outputNode = outputLayer.get(i);
      double output = outputNode.getOutput();
      double der = errorFunc.der(output, targets[i]);
      outputNode.setOutputDer(der);
    }
    //Then conduct backpropagation as usual.
    backProp(targets[0], errorFunc);

  }

  //Note: regularization not yet implemented.
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

  protected static class Edge { //Serializable?
    private Node source;
    private Node destination;
    private Optimizer optimizer;
    private double weight;
    private double accErrorDer;
    private boolean isDead;

    Edge(Node source, Node destination, Optimizer optimizer, double weight) {
      this.source = source;
      this.destination = destination;
      this.optimizer = optimizer;
      this.weight = weight;
      accErrorDer = 0.0;
      isDead = false;
    }

    public boolean isDead() {
      return isDead;
    }

    public void setDead(boolean bool) {
      isDead = bool;
    }


    public double getWeight() {
      return weight;
    }

    public void setWeight(double weight) {
      this.weight = weight;
    }

    public Node getSource() {
      return source;
    }

    public Node getDestination() {
      return destination;
    }

    public Optimizer getOptimizer() {
      return optimizer;
    }

    public void setAccErrorDer(double accErrorDer) {
      this.accErrorDer = accErrorDer;
    }

    public double getAccErrorDer() {
      return accErrorDer;
    }

  }

  protected static class Node {
    private List<Edge> inputEdges;
    private List<Edge> outputEdges;
    private double totalInput;
    private double inputDer;
    private double output;
    private double outputDer;
    private final Activation activation;

    protected Node(Activation activation) {
      this.activation = activation;
      inputEdges = null;
      outputEdges = null;
      totalInput = 0d;
      inputDer = 0d;
      output = 0d;
      outputDer = 0d;
    }

    protected void addInputEdge(Edge edge) {
      if (inputEdges == null)
        inputEdges = new ArrayList<>();
      inputEdges.add(edge);
    }

    protected void addOutputEdge(Edge edge) {
      if (outputEdges == null)
        outputEdges = new ArrayList<>();
      outputEdges.add(edge);
    }

    protected void updateOutput() {
      totalInput = inputEdges.get(0).getWeight(); //The first edge is the bias.

      for (int i = 1; i < inputEdges.size(); i++) {
        Edge edge = inputEdges.get(i);
        totalInput += edge.getSource().getOutput() * edge.getWeight();
      }
      output = activation.output(totalInput);
    }

    public void setOutput(double output) { //This will be used for input layer only.
      this.output = output;
    }

    public double getOutput() {
      return output;
    }

    public void setOutputDer(double outputDer) {
      this.outputDer = outputDer;
    }

    public double getOutputDer() {
      return outputDer;
    }

    public double getInputDer() {
      return inputDer;
    }

    public void setInputDer(double inputDer) {
      this.inputDer = inputDer;
    }

    public Activation getActivation() {
      return activation;
    }

    public double getTotalInput() {
      return totalInput;
    }

    public List<Edge> getInputEdges() {
      return inputEdges;
    }

    public List<Edge> getOutputEdges() {
      return outputEdges;
    }
  }
}
