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

public class MLPTest {

  @Test
  public void testConstructorMinimum() throws Exception {
    /*
     * create 1 x 1 network
     *
     *      \      (bias)
     *   o - o
     *
     */
    MLP mlp = new MLP(1, NetworkShape.parseSetting("1,Sigmoid"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    Assert.assertEquals(2, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    MLP.MNode inputNode = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode.getActivation() instanceof Activation.Identity);
    MLP.MNode hiddenNode = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode.getInputEdges().size());
    Assert.assertEquals(0, hiddenNode.getOutputEdges().size());

    MLP.Edge outputEdge = inputNode.getOutputEdge(0);
    MLP.Edge inputEdge0 = hiddenNode.getInputEdge(0);
    Assert.assertFalse(outputEdge == inputEdge0);
    MLP.Edge inputEdge1 = hiddenNode.getInputEdge(1);
    Assert.assertTrue(outputEdge == inputEdge1);

    Assert.assertTrue(outputEdge.getSource() == inputNode);
    Assert.assertTrue(outputEdge.getDestination() == hiddenNode);
    Assert.assertNull(inputEdge0.getSource());
    Assert.assertTrue(inputEdge0.getDestination() == hiddenNode);
    Assert.assertTrue(inputEdge1.getSource() == inputNode);
    Assert.assertTrue(inputEdge1.getDestination() == hiddenNode);

    Assert.assertTrue(outputEdge.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge0.getOptimizer() instanceof Optimizer.SGD);
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
    MLP mlp = new MLP(2, NetworkShape.parseSetting("2,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L2(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L2);

    Assert.assertEquals(2, mlp.network.size());
    Assert.assertEquals(2, mlp.getLayer(0).size());
    Assert.assertEquals(2, mlp.getLayer(1).size());
    MLP.MNode inputNode0 = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode0.getActivation() instanceof Activation.Identity);
    MLP.MNode inputNode1 = mlp.getNode(0, 1);
    Assert.assertTrue(inputNode1.getActivation() instanceof Activation.Identity);
    MLP.MNode hiddenNode0 = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.ReLU);
    MLP.MNode hiddenNode1 = mlp.getNode(1, 1);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(2, inputNode0.getOutputEdges().size());
    Assert.assertEquals(2, inputNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(0, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(0, hiddenNode1.getOutputEdges().size());

    MLP.Edge outputEdge00 = inputNode0.getOutputEdge(0);
    MLP.Edge outputEdge01 = inputNode0.getOutputEdge(1);
    MLP.Edge outputEdge10 = inputNode1.getOutputEdge(0);
    MLP.Edge outputEdge11 = inputNode1.getOutputEdge(1);
    MLP.Edge inputEdge00 = hiddenNode0.getInputEdge(0);
    MLP.Edge inputEdge01 = hiddenNode0.getInputEdge(1);
    MLP.Edge inputEdge02 = hiddenNode0.getInputEdge(2);
    MLP.Edge inputEdge10 = hiddenNode1.getInputEdge(0);
    MLP.Edge inputEdge11 = hiddenNode1.getInputEdge(1);
    MLP.Edge inputEdge12 = hiddenNode1.getInputEdge(2);
    Assert.assertTrue(outputEdge00 == inputEdge01);
    Assert.assertTrue(outputEdge01 == inputEdge11);
    Assert.assertTrue(outputEdge10 == inputEdge02);
    Assert.assertTrue(outputEdge11 == inputEdge12);

    Assert.assertTrue(outputEdge00.getSource() == inputNode0);
    Assert.assertTrue(outputEdge00.getDestination() == hiddenNode0);
    Assert.assertTrue(outputEdge01.getSource() == inputNode0);
    Assert.assertTrue(outputEdge01.getDestination() == hiddenNode1);
    Assert.assertTrue(outputEdge10.getSource() == inputNode1);
    Assert.assertTrue(outputEdge10.getDestination() == hiddenNode0);
    Assert.assertTrue(outputEdge11.getSource() == inputNode1);
    Assert.assertTrue(outputEdge11.getDestination() == hiddenNode1);

    Assert.assertNull(inputEdge00.getSource());
    Assert.assertNull(inputEdge10.getSource());

    Assert.assertTrue(inputEdge00.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge01.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge02.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge10.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge11.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge12.getOptimizer() instanceof Optimizer.SGD);
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
    MLP mlp = new MLP(1, NetworkShape.parseSetting("1,Sigmoid 1,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(1, mlp.getLayer(0).size());
    Assert.assertEquals(1, mlp.getLayer(1).size());
    Assert.assertEquals(1, mlp.getLayer(2).size());
    MLP.MNode inputNode = mlp.getNode(0, 0);
    Assert.assertTrue(inputNode.getActivation() instanceof Activation.Identity);
    MLP.MNode hiddenNode0 = mlp.getNode(1, 0);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.Sigmoid);
    MLP.MNode hiddenNode1 = mlp.getNode(2, 0);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, inputNode.getInputEdges().size());
    Assert.assertEquals(1, inputNode.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(1, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(0, hiddenNode1.getOutputEdges().size());

    MLP.Edge outputEdge0 = inputNode.getOutputEdge(0);
    MLP.Edge outputEdge1 = hiddenNode0.getOutputEdge(0);
    MLP.Edge inputEdge00 = hiddenNode0.getInputEdge(0);
    MLP.Edge inputEdge01 = hiddenNode0.getInputEdge(1);
    MLP.Edge inputEdge10 = hiddenNode1.getInputEdge(0);
    MLP.Edge inputEdge11 = hiddenNode1.getInputEdge(1);
    Assert.assertTrue(outputEdge0 == inputEdge01);
    Assert.assertTrue(outputEdge1 == inputEdge11);

    Assert.assertTrue(outputEdge0.getSource() == inputNode);
    Assert.assertTrue(outputEdge0.getDestination() == hiddenNode0);
    Assert.assertNull(inputEdge00.getSource());
    Assert.assertTrue(inputEdge00.getDestination() == hiddenNode0);
    Assert.assertTrue(inputEdge01.getSource() == inputNode);
    Assert.assertTrue(inputEdge01.getDestination() == hiddenNode0);
    Assert.assertNull(inputEdge10.getSource());
    Assert.assertTrue(inputEdge10.getDestination() == hiddenNode1);
    Assert.assertTrue(inputEdge11.getSource() == hiddenNode0);
    Assert.assertTrue(inputEdge11.getDestination() == hiddenNode1);

    Assert.assertTrue(inputEdge00.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge01.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge10.getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(inputEdge11.getOptimizer() instanceof Optimizer.SGD);
  }
}
