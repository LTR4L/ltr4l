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

/**
 * Edge (link) in the network.
 * Holds information about which nodes are connected, the weight between the nodes, and dw.
 */
abstract class AbstractEdge<N extends AbstractNode> { //Serializable?
  protected final N source;
  protected final N destination;
  protected final Optimizer optimizer;
  protected double weight;

  AbstractEdge(N source, N destination, Optimizer optimizer, double weight) {
    this.source = source;
    this.destination = destination;
    this.optimizer = optimizer;
    this.weight = weight;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double weight) {
    this.weight = weight;
  }

  public N getSource() {
    return source;
  }

  public N getDestination() {
    return destination;
  }

  public Optimizer getOptimizer() {
    return optimizer;
  }

}
