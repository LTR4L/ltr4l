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

import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;
import org.ltr4l.nn.MLP.*;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;

public class Autoencoder extends AbstractMLP<Autoencoder.AENode, MLP.Edge> implements Encoder {
  private final int eLayerIdx; //Encoded layer index
  private final double sparsity;
  private final double beta;

  public static Autoencoder get(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel, double sparsity, double beta){
    int size = networkShape.size();
    assert(size >= 1);
    for(int i = 0; i < size; i++) {
      NetworkShape.LayerSetting ls = networkShape.getLayerSetting(i);
      if (!(ls.getActivation() == Activation.Type.Sigmoid))
        throw new IllegalArgumentException("Non-Sigmoid Activations are not supported for Autoencoder!");
      if(i == size - 1)
        break; //Don't duplicate encoding layer
      networkShape.add(ls.getNum(), ls.getActivation(), networkShape.size() - i);
    }
    return new Autoencoder(inputDim, networkShape, optFact, regularization, weightModel, sparsity, beta);
  }

  //TODO: Make config for Autoencoder, and get sparsity and beta from config...
  protected Autoencoder(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel, double sparsity, double beta) {
    super(inputDim, networkShape, optFact, regularization, weightModel);
    List<AENode> outputs = new ArrayList<>(); //Output layer should mirror input, but nodes should be different reference
    getLayer(0).forEach(n -> outputs.add(new AENode(Activation.Type.Identity)));
    network.add(outputs);
    eLayerIdx = (network.size() / 2) + 1;
    this.sparsity = sparsity;
    this.beta = beta;
  }

  @Override
  public List<Double> encode(List<Double> features) {
    forwardProp(features);
    return network.get(eLayerIdx).stream().map(AENode::getOutput).collect(Collectors.toCollection(ArrayList::new));
  }

  @Override
  public List<Double> decode(List<Double> features) {
    List<AENode> encodedLayer = network.get(eLayerIdx);
    assert(encodedLayer.size() == features.size());
    resetOutputs();
    for (int i = 0; i < encodedLayer.size(); i++) //Input encoded features
      encodedLayer.get(i).setOutput(features.get(i));
    for (int i = eLayerIdx + 1; i < network.size(); i++){ //Conduct forward prop for half the layer
      List<AENode> layer = network.get(i);
      for (AENode node: layer)
        node.updateOutput();
    }
    return network.get(network.size() - 1).stream().map(AENode::getOutput).collect(Collectors.toCollection(ArrayList::new));
  }

  private void resetOutputs(){
    for(List<AENode> layer : network)
      for(AENode node : layer)
        node.setOutput(0);
  }

  public void backProp(Error errorFunc, double... target) {
    setOutputLayerDerivatives(errorFunc, target);

    for (int layerIdx = network.size() - 1; layerIdx >= 1; layerIdx--) { //When going through each layer, you modify the previous layer.
      List<AENode> layer = network.get(layerIdx);

      for (AENode node : layer) {
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
        List<AENode> previousLayer = network.get(layerIdx - 1);
        for (AENode node : previousLayer) {
          double oder = 0;
          for (Edge outEdge : node.getOutputEdges()) {
            //∂C/∂Oi = ∂Ik/∂Oi * ∂C/∂Ik
            //TODO: incorporate KL-divergence term. AvgAct will be used here.
            AENode dest = (AENode) outEdge.getDestination();
            oder += outEdge.getWeight() * dest.getInputDer() + dest.getActivation().derivative(dest.totalInput) * calcKLDiv(sparsity, calcAvgAct(), beta);
          }
          node.setOutputDer(oder);
        }
      }
    }
    numAccumulatedDer += 1;
  }

  //Note: regularization not yet implemented.
  public void updateWeights(double lrRate, double rgRate) {
    //Update all weights in all edges.
    for (int layerId = 1; layerId < network.size(); layerId++) {  //All Layers
      List<AENode> layer = network.get(layerId);
      for (AENode node : layer) {
        node.resetActivation();
        for (Edge edge : node.getInputEdges()) {
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

  private static double calcKLDiv(double sparsity, double avgAct, double beta) {
    return beta * ((-sparsity / avgAct) + ((1 - sparsity)/(1 - avgAct)));
  }

  private double calcAvgAct() {
    double totalAct = 0d;
    int nodeNum = 0;
    for (int i = 1; i < network.size() - 1; i++){
      for(AENode node : getLayer(i)){
        totalAct += node.getOutput();
        nodeNum++;
      }
    }
    return totalAct / nodeNum;
  }

  @Override
  protected AENode constructNode(Activation activation) {
    return new AENode(activation);
  }

  @Override
  protected Edge constructEdge(AENode source, AENode destination, Optimizer opt, double weight) {
    return new Edge(source, destination, opt, weight);
  }

  @Override
  protected void addOutputs(NetworkShape networkShape) {
    return;
  }

  protected static class AENode extends MNode{
    private double totalAct;
    private int numAct;

    AENode(Activation activation) {
      super(activation);
      totalAct = 0d;
      numAct = 0;
    }

    @Override
    protected void updateOutput(){
      totalInput = inputEdges.get(0).getWeight(); //The first edge is the bias.

      for (int i = 1; i < inputEdges.size(); i++) {
        Edge edge = inputEdges.get(i);
        totalInput += edge.getSource().getOutput() * edge.getWeight();
      }
      output = activation.output(totalInput);
      totalAct += output;
      numAct++;
    }

    public double getAvgAct(){
      return totalAct / numAct;
    }

    public void resetActivation(){
      totalAct = 0d;
      numAct = 0;
    }

  }
}
