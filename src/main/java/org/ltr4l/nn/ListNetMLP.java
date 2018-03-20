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
import java.util.Properties;
import java.util.stream.Collectors;

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

/**
 * Ranker which holds the model based on a Multi-Layer Perceptron network.
 * The overall structure for ListNetMLP is the same as MLP.
 * The main difference lies in the fact that edges keep track of two derivatives instead of one.
 * Thus, the implementation of backpropagation and updateweights is different.
 * TODO: Create a baseMLP class which will extend Ranker and be the parent of ListNetMLP and MLP.
 */
public class ListNetMLP extends Ranker {

  protected final List<List<LNode>> network;
  protected long iter;
  protected double accErrorDer_exSum;
  protected double accErrorDer_ptSum;
  protected final Regularization regularization;

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
    network = new ArrayList<>();

    WeightInitializer weightInit = WeightInitializer.get(weightModel, inputDim, networkShape);

    //Start with constructing the input layer
    List<LNode> currentLayer = new ArrayList<>();
    for (int i = 0; i < inputDim; i++) {
      currentLayer.add(new LNode(new Activation.Identity()));
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
        LNode currentNode = new LNode(activation);
        Optimizer opt = optFact.getOptimizer();
        currentNode.addInputEdge(new LEdge(null, currentNode, opt, bias)); //add bias edge
        currentLayer.add(currentNode);

        for (LNode previousNode : network.get(layerNum)) {    //Note network.get(layerNum) is previous layer!
          LEdge edge = new LEdge(previousNode, currentNode, optFact.getOptimizer(), weightInit.getNextRandomInitialWeight());
          currentNode.addInputEdge(edge);
          previousNode.addOutputEdge(edge);
        }
      }
    }
    network.get(network.size() - 1).get(0).setOutputDer(1); //Set the last node's output derivative to 1
  }

  public List<LNode> getLayer(int i){
    return network.get(i);
  }

  public LNode getNode(int i, int j){
    return getLayer(i).get(j);
  }

  private List<List<List<Double>>> obtainWeights(){
    return network.stream().filter(layer -> layer.get(0).getOutputEdges() != null).map(layer -> layer.stream()
        .map(node -> node.getOutputEdges().stream()
            .map(edge -> edge.getWeight())
            .collect(Collectors.toList()))
        .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  @Override
  public void writeModel(Properties props, String file) {
    try (PrintWriter pw = new PrintWriter(new FileOutputStream(file))) {
      props.store(pw, "Saved model");
      pw.println("model=" + obtainWeights()); //To ensure model gets written at the end.
      //props.setProperty("model", obtainWeights().toString());
      //props.store(pw, "Saved model");

    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  @Override
  public void readModel(String model){
    int dim = 3;
    model = model.substring(dim, model.length() - dim);
    List<Object> modelList = toList(model, dim);
    List<List<List<Double>>> weights = modelList.stream().map(layer -> ((List<List<Double>>) layer)).collect(Collectors.toList());
    for (int layerId = 0; layerId < network.size() - 1; layerId++){ //Do not process last layer
      List<LNode> layer = network.get(layerId);
      for (int nodeId = 0; nodeId < layer.size(); nodeId++){
        LNode node = layer.get(nodeId);
        List<LEdge> outputEdges = node.getOutputEdges();
        for (int edgeId = 0; edgeId < outputEdges.size(); edgeId ++){
          LEdge edge = outputEdges.get(edgeId);
          edge.setWeight(weights.get(layerId).get(nodeId).get(edgeId));
        }
      }
    }
  }

  @Override
  public double predict(List<Double> features){
    return forwardProp(features);
  }

  public double predict(Document doc) {
    return predict(doc.getFeatures());
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
          double oder = 0;
          for (LEdge outEdge : node.getOutputEdges()) {
            //∂C/∂Oi = ∂Ik/∂Oi * ∂C/∂Ik
            oder += outEdge.getWeight() * outEdge.getDestination().getInputDer();
          }
          node.setOutputDer(oder);
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

  /**
   * Difference between LEdge and Edge is the fact that two derivatives are held.
   * See accErrorDerLabel and accErrorDerPredict.
   */
  static class LEdge {
    private final LNode source;
    private final LNode destination;
    private double weight;
    private final Optimizer optimizer;
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

  static class LNode {
    private List<LEdge> inputEdges;
    private List<LEdge> outputEdges;
    private double totalInput;
    private double inputDer;
    private double output;
    private double outputDer;
    private final Activation activation;

    LNode(Activation activation) {
      this.activation = activation;
      inputEdges = new ArrayList<>();
      outputEdges = new ArrayList<>();
      totalInput = 0d;
      inputDer = 0d;
      output = 0d;
      outputDer = 0d;
    }

    public void addInputEdge(LEdge edge) {
      inputEdges.add(edge);
    }

    public void addOutputEdge(LEdge edge) {
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

    public LEdge getInputEdge(int i){
      return inputEdges.get(i);
    }

    public List<LEdge> getOutputEdges() {
      return outputEdges;
    }

    public LEdge getOutputEdge(int i){
      return outputEdges.get(i);
    }
  }
}
