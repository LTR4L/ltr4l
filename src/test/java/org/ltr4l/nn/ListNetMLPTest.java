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

public class ListNetMLPTest extends MLPTestBase<ListNetMLP.LNode, ListNetMLP.LEdge> {

  @Test
  public void testConstructorMinimum() throws Exception {
    /*
     * create 1 x 1 network
     *
     *      \      (bias)
     *   o - o
     *
     */
    ListNetMLP mlp = new ListNetMLP(1, NetworkShape.parseSetting("1,Sigmoid"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    ListNetMLP.LNode outputNode = mlp.getNode(2, 0);
    Assert.assertTrue(outputNode.getActivation() instanceof Activation.Identity);

    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    ListNetMLP.LNode inputNode = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode.getActivation() instanceof Activation.Identity);
    ListNetMLP.LNode hiddenNode = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode.getOutputEdges().size());
    Assert.assertEquals(2, outputNode.getInputEdges().size());
    Assert.assertEquals(0, outputNode.getOutputEdges().size());

    ListNetMLP.LEdge outputEdge0 = inputNode.getOutputEdge(0);
    ListNetMLP.LEdge inputEdge0 = hiddenNode.getInputEdge(0);
    ListNetMLP.LEdge outputEdge1 = hiddenNode.getOutputEdge(0);
    ListNetMLP.LEdge inputEdge2 = outputNode.getInputEdge(0);
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
    ListNetMLP mlp = new ListNetMLP(2, NetworkShape.parseSetting("2,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L2(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L2);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    ListNetMLP.LNode outputNode = mlp.getNode(2, 0);
    Assert.assertTrue(outputNode.getActivation() instanceof Activation.Identity);

    Assert.assertEquals(2, mlp.getLayer(0).size());
    Assert.assertEquals(2, mlp.getLayer(1).size());
    ListNetMLP.LNode inputNode0 = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode0.getActivation() instanceof Activation.Identity);
    ListNetMLP.LNode inputNode1 = mlp.getNode(0, 1);
    Assert.assertTrue(inputNode1.getActivation() instanceof Activation.Identity);
    ListNetMLP.LNode hiddenNode0 = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.ReLU);
    ListNetMLP.LNode hiddenNode1 = mlp.getNode(1, 1);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(2, inputNode0.getOutputEdges().size());
    Assert.assertEquals(2, inputNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(1, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(3, outputNode.getInputEdges().size());

    ListNetMLP.LEdge inputEdge00 = hiddenNode0.getInputEdge(0);
    ListNetMLP.LEdge inputEdge01 = hiddenNode0.getInputEdge(1);
    ListNetMLP.LEdge inputEdge02 = hiddenNode0.getInputEdge(2);
    ListNetMLP.LEdge inputEdge10 = hiddenNode1.getInputEdge(0);
    ListNetMLP.LEdge inputEdge11 = hiddenNode1.getInputEdge(1);
    ListNetMLP.LEdge inputEdge12 = hiddenNode1.getInputEdge(2);
    ListNetMLP.LEdge inputEdge20 = outputNode.getInputEdge(0);
    ListNetMLP.LEdge inputEdge21 = outputNode.getInputEdge(1);
    ListNetMLP.LEdge inputEdge22 = outputNode.getInputEdge(2);
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
    ListNetMLP mlp = new ListNetMLP(1, NetworkShape.parseSetting("1,Sigmoid 1,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // ListNetMLP always adds an output MNode with Activation.Identity
    Assert.assertEquals(4, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(3).size());
    ListNetMLP.LNode outputNode = mlp.getNode(3, 0);
    Assert.assertTrue(outputNode.getActivation() instanceof Activation.Identity);

    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    ListNetMLP.LNode inputNode = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode.getActivation() instanceof Activation.Identity);
    ListNetMLP.LNode hiddenNode0 = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.Sigmoid);
    ListNetMLP.LNode hiddenNode1 = mlp.getNode(2, 0);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(2, outputNode.getInputEdges().size());
    Assert.assertEquals(0, outputNode.getOutputEdges().size());

    ListNetMLP.LEdge outputEdge0 = inputNode.getOutputEdge(0);
    ListNetMLP.LEdge outputEdge1 = hiddenNode0.getOutputEdge(0);
    ListNetMLP.LEdge inputEdge00 = hiddenNode0.getInputEdge(0);
    ListNetMLP.LEdge inputEdge01 = hiddenNode0.getInputEdge(1);
    ListNetMLP.LEdge inputEdge10 = hiddenNode1.getInputEdge(0);
    ListNetMLP.LEdge inputEdge11 = hiddenNode1.getInputEdge(1);
    ListNetMLP.LEdge outputEdge2 = hiddenNode1.getOutputEdge(0);
    ListNetMLP.LEdge inputEdge20 = outputNode.getInputEdge(0);
    ListNetMLP.LEdge inputEdge21 = outputNode.getInputEdge(1);
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
