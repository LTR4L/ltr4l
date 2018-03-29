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
public class ListNetMLP extends AbstractMLP<ListNetMLP.LNode, ListNetMLP.LEdge> {

  protected double accErrorDer_exSum;
  protected double accErrorDer_ptSum;

  //CONSTRUCT NETWORK
  public ListNetMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(inputDim, networkShape, optFact, regularization, weightModel);
    accErrorDer_exSum = 0;
    accErrorDer_ptSum = 0;
    network.get(network.size() - 1).get(0).setOutputDer(1); //Set the last node's output derivative to 1
  }

  protected void addOutputs(NetworkShape ns){
    ns.add(1, new Activation.Identity());
  }

  @Override
  protected LNode constructNode(Activation activation) {
    return new LNode(activation);
  }

  @Override
  protected LEdge constructEdge(LNode source, LNode destination, Optimizer opt, double weight) {
    return new LEdge(source, destination, opt, weight);
  }

  public double predict(Document doc) {
    return predict(doc.getFeatures());
  }

  //Overloaded to accept Document object as well.
  public double forwardProp(Document doc) {
    List<Double> features = doc.getFeatures();
    return forwardProp(features);

  }

  public void backProp(double target) {
    backProp(null, target);
  }

  //This is for one output node.
  public void backProp(Error errorFunc, double... target) {
    LNode outputNode = network.get(network.size() - 1).get(0);
    double output = outputNode.getOutput();

    double expOutput = Math.exp(output);
    double expTarget = Math.exp(target[0]);

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
  static class LEdge extends AbstractEdge.AbstractFFEdge<LNode> {
    private double accErrorDerLabel;   //Σexp(y)∂f/∂w
    private double accErrorDerPredict; //Σexp(f(x))∂f/∂w

    LEdge(LNode source, LNode destination, Optimizer optimizer, double weight) {
      super(source, destination, optimizer, weight);
      accErrorDerLabel = 0.0;
      accErrorDerPredict = 0.0;
    }

    protected double getAccErrorDerLabel() {
      return accErrorDerLabel;
    }

    protected void setAccErrorDerLabel(double errDer) {
      accErrorDerLabel = errDer;
    }

    protected double getAccErrorDerPredict() {
      return accErrorDerPredict;
    }

    protected void setAccErrorDerPredict(double errDer) {
      accErrorDerPredict = errDer;
    }
  }

  /**
   * MNode which holds LEdges.
   */
  static class LNode extends AbstractNode.Node<LEdge> {

    LNode(Activation activation) {
      super(activation);
    }
  }
}
