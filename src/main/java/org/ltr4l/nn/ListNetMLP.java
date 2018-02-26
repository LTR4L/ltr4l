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

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.stream.Collectors;

import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

public class ListNetMLP {

  protected List<List<LNode>> network;
  protected long iter;
  protected double accErrorDer_exSum;
  protected double accErrorDer_ptSum;
  protected int nWeights;
  protected final Regularization regularization;
  protected static final String DEFAULT_MODEL_FILE = "model.txt";

  //CONSTRUCT NETWORK
  public ListNetMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    //Network shape describes number of nodes and their activation. Example:
    //[
    //[12, Sigmoid ]
    //[4 , Softmax ]
    //[1 , Identity]
    //]
    iter = 1;
    accErrorDer_exSum = 0;
    accErrorDer_ptSum = 0;
    this.regularization = regularization;
    nWeights = inputDim * networkShape.getLayerSetting(0).getNum();  //Number of weights used for Xavier initialization.
    network = new ArrayList<>();

    for (int i = 1; i < networkShape.size(); i++) {
      nWeights += networkShape.getLayerSetting(i - 1).getNum() * networkShape.getLayerSetting(i).getNum();
    }

    //Start with constructing the input layer
    List<LNode> currentLayer = new ArrayList<>();
    for (int i = 0; i < inputDim; i++) {
      currentLayer.add(new LNode(new Activation.Identity()));
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
        LNode currentNode = new LNode(activation);
        Optimizer opt = optFact.getOptimizer();
        currentNode.addInputEdge(new LEdge(null, currentNode, opt, bias)); //add bias edge
        currentLayer.add(currentNode);

        for (LNode previousNode : network.get(layerNum)) {    //Note network.get(layerNum) is previous layer!
          LEdge edge = new LEdge(previousNode, currentNode, optFact.getOptimizer(), weightInit(weightModel));
          currentNode.addInputEdge(edge);
          previousNode.addOutputEdge(edge);
        }
      }
    }
    network.get(network.size() - 1).get(0).setOutputDer(1); //Set the last node's output derivative to 1
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

  private List<List<List<Double>>> obtainWeights(){
    return network.stream().filter(layer -> layer.get(0).getOutputEdges() != null).map(layer -> layer.stream()
        .map(node -> node.getOutputEdges().stream()
            .map(edge -> edge.getWeight())
            .collect(Collectors.toList()))
        .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  public void writeModel() {
    writeModel(DEFAULT_MODEL_FILE);
  }

  public void writeModel(String file){
    try (PrintWriter pw = new PrintWriter(new FileOutputStream(file))) {
      pw.println(obtainWeights());

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public double predict(Document doc) {
    return forwardProp(doc);
  }

  //Feed forward propagation
  //This is for the case of the output layer with one node.
  public double forwardProp(List<Double> features) {
    //Feed document features into input layer.
    List<LNode> layer = network.get(0); //First layer
    for (int i = 0; i < layer.size(); i++) {
      LNode node = layer.get(i);
      node.setOutput(features.get(i));
    }

    //Go through the rest of the layers and update the output.
    for (int layerId = 1; layerId < network.size(); layerId++) {
      layer = network.get(layerId);
      for (LNode node : layer) {
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

  public void backProp(double target) {
    backProp(target, null);
  }

  //This is for one output node.
  public void backProp(double target, Error errorFunc) {
    LNode outputNode = network.get(network.size() - 1).get(0);
    double output = outputNode.getOutput();

    double expOutput = Math.exp(output);
    double expTarget = Math.exp(target);

    //Σexp(f(x)) and Σexp(py)
    accErrorDer_exSum += expOutput;
    accErrorDer_ptSum += expTarget;

    //Don't set error derivative, as it is essentially covered by the accErrorDers
    //First, get the derivative ∂C/∂O and set it to output derivative of the final node.
    //double der = errorFunc.der(output, target);
    //outputNode.setOutputDer(der);

    for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) { //When going through each layer, you modify the previous layer.
      List<LNode> layer = network.get(layerIdx);

      for (LNode node : layer) {
        // Second, find ∂C/∂I by (∂C/∂O)(∂O/∂I)
        // I = total Input; O = output = Activation(I)
        double totalInput = node.getTotalInput();
        double inDer = node.getActivation().derivative(totalInput) * node.getOutputDer();
        node.setInputDer(inDer);

        //First edge is bias.
        LEdge edge = node.getInputEdges().get(0);

        double accErrDer = edge.getAccErrorDerLabel();
        accErrDer += node.getInputDer();
        edge.setAccErrorDerLabel(accErrDer);

        for (int edgeNum = 1; edgeNum < node.getInputEdges().size(); edgeNum++) {
          edge = node.getInputEdges().get(edgeNum);
          if (!edge.isDead()) {
            //(∂C/∂I)*(∂I/∂w) = Σ∂C/∂Ii *(∂Ii/∂w) = ∂C/∂w
            //(∂Ii/∂w) = Oi, because Oi*wi = Ii
            double errorDer = node.getInputDer() * edge.getSource().getOutput(); //(∂C/∂I)*(∂I/∂w)
            accErrDer = edge.getAccErrorDerLabel();
            accErrDer += errorDer * expTarget;
            edge.setAccErrorDerLabel(accErrDer);

            accErrDer = edge.getAccErrorDerPredict();
            accErrDer += errorDer * expOutput;
            edge.setAccErrorDerPredict(accErrDer);
          }
        }
      }
      if (layerIdx != 1) {
        List<LNode> previousLayer = network.get(layerIdx - 1);
        for (LNode node : previousLayer) {
          //double oder = node.getOutputDer();
          //node.setOutputDer(0);
          double oder = 0;
          for (LEdge outEdge : node.getOutputEdges()) {
            //And finally, ∂C/∂w = ∂C/∂I * ∂I/∂w
            oder += outEdge.getWeight() * outEdge.getDestination().getInputDer();
            node.setOutputDer(oder);
          }
        }
      }
    }
    //numAccumulatedDer += 1;
  }

  //Note: regularization not yet implemented.
  public void updateWeights(double lrRate, double rgRate) {
    //Update all weights in all edges.
    for (int layerId = 1; layerId < network.size(); layerId++) {  //All Layers
      List<LNode> layer = network.get(layerId);
      for (LNode node : layer) {                     //All Nodes
        for (LEdge edge : node.getInputEdges()) { //All edges for each node.
          if (!edge.isDead()) {
            double rgDer = 0;
            if (regularization != null)
              rgDer = regularization.derivative(edge.getWeight());

            double accErrDer = -edge.getAccErrorDerLabel() / accErrorDer_ptSum;
            accErrDer += edge.getAccErrorDerPredict() / accErrorDer_exSum;

            double cost = edge.getOptimizer().optimize(accErrDer, lrRate, iter);
            double weight = edge.getWeight();
            weight += cost;

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
            edge.setAccErrorDerPredict(0); //After the change has been applied, reset the accumulated derivatives.
            edge.setAccErrorDerLabel(0);

          }
        }
      }
    }
    accErrorDer_exSum = 0;
    accErrorDer_ptSum = 0;  //Weights updated, now no derivatives have been accumulated.
    iter++;
  }

  public List<List<List<Double>>> getWeights() {
    //Note: went with collect as it is necessary to get a list of all weights anyway.
    return network.stream().filter(layer -> layer.get(0).getOutputEdges() != null).map(layer -> layer.stream()
        .map(node -> node.getOutputEdges().stream()
            .map(edge -> edge.getWeight())
            .collect(Collectors.toList()))
        .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  private static class LEdge {
    private LNode source;
    private LNode destination;
    private double weight;
    private Optimizer optimizer;
    private double accErrorDerLabel;   //Σexp(y)∂f/∂w
    private double accErrorDerPredict; //Σexp(f(x))∂f/∂w
    boolean isDead;

    LEdge(LNode source, LNode destination, Optimizer optimizer, double weight) {
      this.source = source;
      this.destination = destination;
      this.optimizer = optimizer;
      this.weight = weight;
      accErrorDerLabel = 0.0;
      accErrorDerPredict = 0.0;
      isDead = false;
    }

    public Optimizer getOptimizer() {
      return optimizer;
    }

    public double getWeight() {
      return weight;
    }

    public void setWeight(double weight) {
      this.weight = weight;
    }

    public LNode getSource() {
      return source;
    }

    public LNode getDestination() {
      return destination;
    }

    public double getAccErrorDerLabel() {
      return accErrorDerLabel;
    }

    public void setAccErrorDerLabel(double errDer) {
      accErrorDerLabel = errDer;
    }

    public double getAccErrorDerPredict() {
      return accErrorDerPredict;
    }

    public void setAccErrorDerPredict(double errDer) {
      accErrorDerPredict = errDer;
    }

    public boolean isDead() {
      return isDead;
    }

    public void setDead(boolean isDead) {
      this.isDead = isDead;
    }
  }

  private static class LNode {
    private List<LEdge> inputEdges;
    private List<LEdge> outputEdges;
    private double totalInput;
    private double inputDer;
    private double output;
    private double outputDer;
    private Activation activation;

    LNode(Activation activation) {
      this.activation = activation;
      inputEdges = null;
      outputEdges = null;
      totalInput = 0d;
      inputDer = 0d;
      output = 0d;
      outputDer = 0d;
    }

    public void addInputEdge(LEdge edge) {
      if (inputEdges == null)
        inputEdges = new ArrayList<>();
      inputEdges.add(edge);
    }

    public void addOutputEdge(LEdge edge) {
      if (outputEdges == null)
        outputEdges = new ArrayList<>();
      outputEdges.add(edge);
    }

    public void updateOutput() {
      totalInput = inputEdges.get(0).getWeight(); //The first edge is the bias.

      for (int i = 1; i < inputEdges.size(); i++) {
        LEdge edge = inputEdges.get(i);
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

    public List<LEdge> getInputEdges() {
      return inputEdges;
    }

    public List<LEdge> getOutputEdges() {
      return outputEdges;
    }

  }
}
