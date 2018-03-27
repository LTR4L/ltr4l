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

/**
 * MNode in the network.
 * Holds information regarding the edges (which nodes are connected), and based on that information
 * the total input and output. Also contains information about Activation.
 */
public abstract class AbstractNode<E extends AbstractEdge> {
  protected final List<E> inputEdges;
  protected final List<E> outputEdges;
  protected double totalInput;
  protected double inputDer;
  protected double output;
  protected double outputDer;
  protected final Activation activation;

  protected AbstractNode(Activation activation) {
    this.activation = activation;
    inputEdges = new ArrayList<>();
    outputEdges = new ArrayList<>();
    totalInput = 0d;
    inputDer = 0d;
    output = 0d;
    outputDer = 0d;
  }

  protected void addInputEdge(E edge) {
    inputEdges.add(edge);
  }

  protected void addOutputEdge(E edge) {
    outputEdges.add(edge);
  }

  protected abstract void updateOutput();

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

  public List<E> getInputEdges() {
    return inputEdges;
  }

  public List<E> getOutputEdges() {
    return outputEdges;
  }

  public E getInputEdge(int i) { return inputEdges.get(i);}

  public E getOutputEdge(int i) { return outputEdges.get(i);}

  /**
   * This is the Node for FFN (Feed-Forward Networks).
   * @param <E1>
   */
  public static class Node<E1 extends AbstractEdge.AbstractFFEdge> extends AbstractNode<E1>{
    Node(Activation activation){
      super(activation);
    }

    @Override
    protected void updateOutput(){
      totalInput = inputEdges.get(0).getWeight(); //The first edge is the bias.

      for (int i = 1; i < inputEdges.size(); i++) {
        E1 edge = inputEdges.get(i);
        totalInput += edge.getSource().getOutput() * edge.getWeight();
      }
      output = activation.output(totalInput);
    }
  }

}
