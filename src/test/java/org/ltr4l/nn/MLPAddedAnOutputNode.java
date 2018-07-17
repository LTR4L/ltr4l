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

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.tools.Regularization;
import org.ltr4l.nn.AbstractNode.Node;
import org.ltr4l.nn.AbstractEdge.AbstractFFEdge;

public abstract class MLPAddedAnOutputNode<N extends Node, E extends AbstractFFEdge, MLP extends AbstractMLP> extends MLPTestBase {

  protected abstract MLP create(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel);
  
  protected Node getNode(AbstractMLP mlp, int i, int j){ return (Node) mlp.getNode(i, j); }
  protected AbstractFFEdge getOutputEdge(Node node, int i){ return (AbstractFFEdge) node.getOutputEdge(i); }
  protected AbstractFFEdge getInputEdge(Node node, int i){ return (AbstractFFEdge) node.getInputEdge(i); }

  @Test
  public void testConstructorMinimum() throws Exception {
    /*
     * create 1 x 1 network
     *
     *      \      (bias)
     *   o - o
     *
     */
    AbstractMLP mlp = create(1, NetworkShape.parseSetting("1,Sigmoid"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    Node outputNode = getNode(mlp, 2, 0);
    Assert.assertEquals(outputNode.getActivation(), Activation.Type.Identity);

    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    Node inputNode = getNode(mlp, 0, 0);
    Assert.assertTrue(inputNode.getActivation() == Activation.Type.Identity);
    Node hiddenNode = getNode(mlp, 1, 0);
    Assert.assertEquals(hiddenNode.getActivation(), Activation.Type.Sigmoid);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode.getOutputEdges().size());
    Assert.assertEquals(2, outputNode.getInputEdges().size());
    Assert.assertEquals(0, outputNode.getOutputEdges().size());

    AbstractFFEdge outputEdge0 = getOutputEdge(inputNode,0);
    AbstractFFEdge inputEdge0 = getInputEdge(hiddenNode, 0);
    AbstractFFEdge outputEdge1 = getOutputEdge(hiddenNode, 0);
    AbstractFFEdge inputEdge2 = getInputEdge(outputNode, 0);
    assertBetweenNodes(inputNode, 0, hiddenNode, 1);
    assertBetweenNodes(hiddenNode, 0, outputNode, 1);
    assertBiasEdge(inputEdge0, hiddenNode);
    assertBiasEdge(inputEdge2, outputNode);

    Assert.assertTrue(outputEdge0.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge0.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(outputEdge1.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge2.getOptimizer() instanceof Optimizer.SGD);
  }

  @Test
  public void testConstructor2x2() throws Exception {
    /*
     * create 2 x 2 network
     *
     *      \      (bias)
     *   o - o
     *     X\      (bias)
     *   o - o
     *
     */
    AbstractMLP mlp = create(2, NetworkShape.parseSetting("2,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L2(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L2);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    Node outputNode = getNode(mlp, 2, 0);
    Assert.assertEquals(outputNode.getActivation(), Activation.Type.Identity);

    Assert.assertEquals(2, mlp.getLayer(0).size());
    Assert.assertEquals(2, mlp.getLayer(1).size());
    Node inputNode0 = getNode(mlp, 0, 0);
    Assert.assertEquals(inputNode0.getActivation(), Activation.Type.Identity);
    Node inputNode1 = getNode(mlp, 0, 1);
    Assert.assertEquals(inputNode1.getActivation(), Activation.Type.Identity);
    Node hiddenNode0 = getNode(mlp, 1, 0);
    Assert.assertEquals(hiddenNode0.getActivation(), Activation.Type.ReLU);
    Node hiddenNode1 = getNode(mlp, 1, 1);
    Assert.assertEquals(hiddenNode1.getActivation(), Activation.Type.ReLU);
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(2, inputNode0.getOutputEdges().size());
    Assert.assertEquals(2, inputNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(1, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(3, outputNode.getInputEdges().size());


    AbstractFFEdge inputEdge00 = getInputEdge(hiddenNode0, 0);
    AbstractFFEdge inputEdge01 = getInputEdge(hiddenNode0, 1);
    AbstractFFEdge inputEdge02 = getInputEdge(hiddenNode0, 2);
    AbstractFFEdge inputEdge10 = getInputEdge(hiddenNode1, 0);
    AbstractFFEdge inputEdge11 = getInputEdge(hiddenNode1, 1);
    AbstractFFEdge inputEdge12 = getInputEdge(hiddenNode1, 2);
    AbstractFFEdge inputEdge20 = getInputEdge(outputNode, 0);
    AbstractFFEdge inputEdge21 = getInputEdge(outputNode, 1);
    AbstractFFEdge inputEdge22 = getInputEdge(outputNode, 2);
    assertBetweenNodes(inputNode0, 0, hiddenNode0, 1);
    assertBetweenNodes(inputNode0, 1, hiddenNode1, 1);
    assertBetweenNodes(inputNode1, 0, hiddenNode0, 2);
    assertBetweenNodes(inputNode1, 1, hiddenNode1, 2);
    assertBetweenNodes(hiddenNode0, 0, outputNode, 1);
    assertBetweenNodes(hiddenNode1, 0, outputNode, 2);
    assertBiasEdge(inputEdge00, hiddenNode0);
    assertBiasEdge(inputEdge10, hiddenNode1);
    assertBiasEdge(inputEdge20, outputNode);

    Assert.assertTrue(inputEdge00.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge01.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge02.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge10.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge11.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge12.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge20.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge21.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge22.getOptimizer() instanceof Optimizer.SGD);
  }

  @Test
  public void testConstructor1x1x1() throws Exception {
    /*
     * create 1 x 1 x 1 network
     *
     *      \   \   (bias)
     *   o - o - o
     *
     */
    AbstractMLP mlp = create(1, NetworkShape.parseSetting("1,Sigmoid 1,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(4, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(3).size());
    Node outputNode = getNode(mlp, 3, 0);
    Assert.assertEquals(outputNode.getActivation(), Activation.Type.Identity);

    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    Node inputNode = getNode(mlp, 0, 0);
    Assert.assertEquals(inputNode.getActivation(), Activation.Type.Identity);
    Node hiddenNode0 = getNode(mlp, 1, 0);
    Assert.assertEquals(hiddenNode0.getActivation(), Activation.Type.Sigmoid);
    Node hiddenNode1 = getNode(mlp, 2, 0);
    Assert.assertEquals(hiddenNode1.getActivation(), Activation.Type.ReLU);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(2, outputNode.getInputEdges().size());
    Assert.assertEquals(0, outputNode.getOutputEdges().size());

    AbstractFFEdge outputEdge0 = getOutputEdge(inputNode, 0);
    AbstractFFEdge outputEdge1 = getOutputEdge(hiddenNode0, 0);
    AbstractFFEdge inputEdge00 = getInputEdge(hiddenNode0, 0);
    AbstractFFEdge inputEdge01 = getInputEdge(hiddenNode0, 1);
    AbstractFFEdge inputEdge10 = getInputEdge(hiddenNode1, 0);
    AbstractFFEdge inputEdge11 = getInputEdge(hiddenNode1, 1);
    AbstractFFEdge outputEdge2 = getOutputEdge(hiddenNode1,0);
    AbstractFFEdge inputEdge20 = getInputEdge(outputNode, 0);
    AbstractFFEdge inputEdge21 = getInputEdge(outputNode, 1);
    assertBetweenNodes(inputNode, 0, hiddenNode0, 1);
    assertBetweenNodes(hiddenNode0, 0, hiddenNode1, 1);
    assertBetweenNodes(hiddenNode1, 0, outputNode, 1);
    assertBiasEdge(inputEdge00, hiddenNode0);
    assertBiasEdge(inputEdge10, hiddenNode1);
    assertBiasEdge(inputEdge20, outputNode);

    Assert.assertTrue(inputEdge00.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge01.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge10.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge11.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge20.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge21.getOptimizer() instanceof Optimizer.SGD);
  }
}
