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

public class SortNetMLPTest {

  @Test
  public void testConstructorMinimum() throws Exception {
    /*
     * create 1 x 1 network
     *
     *      \      (bias)
     *   o - o
     *
     */
    SortNetMLP mlp = new SortNetMLP(1, NetworkShape.parseSetting("1,Sigmoid"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // SortNetMLP always adds an output Node with Activation.Sigmoid
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(2, mlp.getLayer(2).size());
    SortNetMLP.SNode outputNode0 = mlp.getNode(2, 0);
    SortNetMLP.SNode outputNode1 = mlp.getNode(2, 1);
    Assert.assertTrue(outputNode0.getActivation() instanceof Activation.Sigmoid);
    Assert.assertTrue(outputNode1.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, outputNode0.getGroup());
    Assert.assertEquals(1, outputNode1.getGroup());

    Assert.assertEquals(2, mlp.getLayer(0).size());
    Assert.assertEquals(2, mlp.getLayer(1).size());
    SortNetMLP.SNode inputNode0 = mlp.getNode(0, 0);
    SortNetMLP.SNode inputNode1 = mlp.getNode(0, 1);
    Assert.assertTrue(inputNode0.getActivation() instanceof Activation.Identity);
    Assert.assertTrue(inputNode1.getActivation() instanceof Activation.Identity);
    Assert.assertEquals(0, inputNode0.getGroup());
    Assert.assertEquals(1, inputNode1.getGroup());
    SortNetMLP.SNode hiddenNode0 = mlp.getNode(1, 0);
    SortNetMLP.SNode hiddenNode1 = mlp.getNode(1, 1);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.Sigmoid);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, hiddenNode0.getGroup());
    Assert.assertEquals(1, hiddenNode1.getGroup());
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(2, inputNode0.getOutputEdges().size());
    Assert.assertEquals(2, inputNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(3, outputNode0.getInputEdges().size());
    Assert.assertEquals(3, outputNode1.getInputEdges().size());
    Assert.assertEquals(0, outputNode0.getOutputEdges().size());
    Assert.assertEquals(0, outputNode1.getOutputEdges().size());

    SortNetMLP.SEdge outputEdge00 = inputNode0.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge10 = inputNode0.getOutputEdge(1);
    SortNetMLP.SEdge outputEdge01 = inputNode1.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge11 = inputNode1.getOutputEdge(1);
    SortNetMLP.SEdge inputEdge00 = hiddenNode0.getInputEdge(0);
    SortNetMLP.SEdge inputEdge01 = hiddenNode0.getInputEdge(1);
    SortNetMLP.SEdge inputEdge02 = hiddenNode0.getInputEdge(2);
    SortNetMLP.SEdge inputEdge10 = hiddenNode1.getInputEdge(0);
    SortNetMLP.SEdge inputEdge11 = hiddenNode1.getInputEdge(1);
    SortNetMLP.SEdge inputEdge12 = hiddenNode1.getInputEdge(2);
    assertEquals4(outputEdge00, outputEdge01, inputEdge01, inputEdge11);
    assertEquals4(outputEdge10, outputEdge11, inputEdge02, inputEdge12);
    Assert.assertTrue(inputEdge00 == inputEdge10);
    SortNetMLP.SEdge outputEdge20 = hiddenNode0.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge30 = hiddenNode0.getOutputEdge(1);
    SortNetMLP.SEdge outputEdge21 = hiddenNode1.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge31 = hiddenNode1.getOutputEdge(1);
    SortNetMLP.SEdge inputEdge20 = outputNode0.getInputEdge(0);
    SortNetMLP.SEdge inputEdge21 = outputNode0.getInputEdge(1);
    SortNetMLP.SEdge inputEdge22 = outputNode0.getInputEdge(2);
    SortNetMLP.SEdge inputEdge30 = outputNode1.getInputEdge(0);
    SortNetMLP.SEdge inputEdge31 = outputNode1.getInputEdge(1);
    SortNetMLP.SEdge inputEdge32 = outputNode1.getInputEdge(2);
    assertEquals4(outputEdge20, outputEdge21, inputEdge21, inputEdge31);
    assertEquals4(outputEdge30, outputEdge31, inputEdge22, inputEdge32);
    Assert.assertTrue(inputEdge20 == inputEdge30);

    assertBetweenPairs(inputNode0, inputNode1, outputEdge00, hiddenNode0, hiddenNode1);
    assertBetweenPairs(inputNode1, inputNode0, outputEdge10, hiddenNode0, hiddenNode1);
    assertBiasEdge(inputEdge00, hiddenNode0, hiddenNode1);

    assertBetweenPairs(hiddenNode0, hiddenNode1, outputEdge20, outputNode0, outputNode1);
    assertBetweenPairs(hiddenNode1, hiddenNode0, outputEdge30, outputNode0, outputNode1);
    assertBiasEdge(inputEdge20, outputNode0, outputNode1);
  }

  private void assertEquals4(SortNetMLP.SEdge oe0, SortNetMLP.SEdge oe1, SortNetMLP.SEdge ie0, SortNetMLP.SEdge ie1) throws Exception {
    Assert.assertTrue(oe0 == oe1);
    Assert.assertTrue(oe0 == ie0);
    Assert.assertTrue(oe0 == ie1);
  }

  private void assertBetweenPairs(SortNetMLP.SNode sn0, SortNetMLP.SNode sn1, SortNetMLP.SEdge edge,
                                  SortNetMLP.SNode dn0, SortNetMLP.SNode dn1) throws Exception {
    Assert.assertEquals(2, edge.getSource().length);
    Assert.assertTrue(edge.getSource()[0] == sn0);
    Assert.assertTrue(edge.getSource()[1] == sn1);
    Assert.assertEquals(2, edge.getDestination().length);
    Assert.assertTrue(edge.getDestination()[0] == dn0);
    Assert.assertTrue(edge.getDestination()[1] == dn1);
  }

  private void assertBiasEdge(SortNetMLP.SEdge edge, SortNetMLP.SNode dn0, SortNetMLP.SNode dn1) throws Exception {
    Assert.assertNull(edge.getSource());
    Assert.assertEquals(2, edge.getDestination().length);
    Assert.assertTrue(edge.getDestination()[0] == dn0);
    Assert.assertTrue(edge.getDestination()[1] == dn1);
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
    SortNetMLP mlp = new SortNetMLP(2, NetworkShape.parseSetting("2,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L2(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L2);

    // SortNetMLP always adds an output Node with Activation.Sigmoid
    Assert.assertEquals(3, mlp.network.size());
    Assert.assertEquals(2, mlp.getLayer(2).size());
    SortNetMLP.SNode outputNode0 = mlp.getNode(2, 0);
    SortNetMLP.SNode outputNode1 = mlp.getNode(2, 1);
    Assert.assertTrue(outputNode0.getActivation() instanceof Activation.Sigmoid);
    Assert.assertTrue(outputNode1.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, outputNode0.getGroup());
    Assert.assertEquals(1, outputNode1.getGroup());

    Assert.assertEquals(4, mlp.getLayer(0).size());
    Assert.assertEquals(4, mlp.getLayer(1).size());
    SortNetMLP.SNode inputNode0 = mlp.getNode(0, 0);
    SortNetMLP.SNode inputNode1 = mlp.getNode(0, 1);
    SortNetMLP.SNode inputNode2 = mlp.getNode(0, 2);
    SortNetMLP.SNode inputNode3 = mlp.getNode(0, 3);
    Assert.assertTrue(inputNode0.getActivation() instanceof Activation.Identity);
    Assert.assertTrue(inputNode1.getActivation() instanceof Activation.Identity);
    Assert.assertTrue(inputNode2.getActivation() instanceof Activation.Identity);
    Assert.assertTrue(inputNode3.getActivation() instanceof Activation.Identity);
    Assert.assertEquals(0, inputNode0.getGroup());
    Assert.assertEquals(0, inputNode1.getGroup());
    Assert.assertEquals(1, inputNode2.getGroup());
    Assert.assertEquals(1, inputNode3.getGroup());
    SortNetMLP.SNode hiddenNode0 = mlp.getNode(1, 0);
    SortNetMLP.SNode hiddenNode1 = mlp.getNode(1, 1);
    SortNetMLP.SNode hiddenNode2 = mlp.getNode(1, 2);
    SortNetMLP.SNode hiddenNode3 = mlp.getNode(1, 3);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.ReLU);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.ReLU);
    Assert.assertTrue(hiddenNode2.getActivation() instanceof Activation.ReLU);
    Assert.assertTrue(hiddenNode3.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, hiddenNode0.getGroup());
    Assert.assertEquals(0, hiddenNode1.getGroup());
    Assert.assertEquals(1, hiddenNode2.getGroup());
    Assert.assertEquals(1, hiddenNode3.getGroup());
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(0, inputNode2.getInputEdges().size());
    Assert.assertEquals(0, inputNode3.getInputEdges().size());
    Assert.assertEquals(4, inputNode0.getOutputEdges().size());
    Assert.assertEquals(4, inputNode1.getOutputEdges().size());
    Assert.assertEquals(4, inputNode2.getOutputEdges().size());
    Assert.assertEquals(4, inputNode3.getOutputEdges().size());
    Assert.assertEquals(5, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(5, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(5, hiddenNode2.getInputEdges().size());
    Assert.assertEquals(5, hiddenNode3.getInputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode2.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode3.getOutputEdges().size());
    Assert.assertEquals(5, outputNode0.getInputEdges().size());
    Assert.assertEquals(5, outputNode1.getInputEdges().size());
    Assert.assertEquals(0, outputNode0.getOutputEdges().size());
    Assert.assertEquals(0, outputNode1.getOutputEdges().size());

    SortNetMLP.SEdge oe00 = inputNode0.getOutputEdge(0);
    SortNetMLP.SEdge oe10 = inputNode1.getOutputEdge(0);
    SortNetMLP.SEdge oe20 = inputNode2.getOutputEdge(0);
    SortNetMLP.SEdge oe30 = inputNode3.getOutputEdge(0);
    SortNetMLP.SEdge oe01 = inputNode0.getOutputEdge(1);
    SortNetMLP.SEdge oe11 = inputNode1.getOutputEdge(1);
    SortNetMLP.SEdge oe21 = inputNode2.getOutputEdge(1);
    SortNetMLP.SEdge oe31 = inputNode3.getOutputEdge(1);
    SortNetMLP.SEdge oe02 = inputNode0.getOutputEdge(2);
    SortNetMLP.SEdge oe12 = inputNode1.getOutputEdge(2);
    SortNetMLP.SEdge oe22 = inputNode2.getOutputEdge(2);
    SortNetMLP.SEdge oe32 = inputNode3.getOutputEdge(2);
    SortNetMLP.SEdge oe03 = inputNode0.getOutputEdge(3);
    SortNetMLP.SEdge oe13 = inputNode1.getOutputEdge(3);
    SortNetMLP.SEdge oe23 = inputNode2.getOutputEdge(3);
    SortNetMLP.SEdge oe33 = inputNode3.getOutputEdge(3);
    SortNetMLP.SEdge ie00 = hiddenNode0.getInputEdge(0);
    SortNetMLP.SEdge ie10 = hiddenNode1.getInputEdge(0);
    SortNetMLP.SEdge ie20 = hiddenNode2.getInputEdge(0);
    SortNetMLP.SEdge ie30 = hiddenNode3.getInputEdge(0);
    SortNetMLP.SEdge ie01 = hiddenNode0.getInputEdge(1);
    SortNetMLP.SEdge ie11 = hiddenNode1.getInputEdge(1);
    SortNetMLP.SEdge ie21 = hiddenNode2.getInputEdge(1);
    SortNetMLP.SEdge ie31 = hiddenNode3.getInputEdge(1);
    SortNetMLP.SEdge ie02 = hiddenNode0.getInputEdge(2);
    SortNetMLP.SEdge ie12 = hiddenNode1.getInputEdge(2);
    SortNetMLP.SEdge ie22 = hiddenNode2.getInputEdge(2);
    SortNetMLP.SEdge ie32 = hiddenNode3.getInputEdge(2);
    SortNetMLP.SEdge ie03 = hiddenNode0.getInputEdge(3);
    SortNetMLP.SEdge ie13 = hiddenNode1.getInputEdge(3);
    SortNetMLP.SEdge ie23 = hiddenNode2.getInputEdge(3);
    SortNetMLP.SEdge ie33 = hiddenNode3.getInputEdge(3);
    SortNetMLP.SEdge ie04 = hiddenNode0.getInputEdge(4);
    SortNetMLP.SEdge ie14 = hiddenNode1.getInputEdge(4);
    SortNetMLP.SEdge ie24 = hiddenNode2.getInputEdge(4);
    SortNetMLP.SEdge ie34 = hiddenNode3.getInputEdge(4);
    assertEquals4(oe00, oe20, ie01, ie21);
    assertEquals4(oe01, oe21, ie02, ie22);
    assertEquals4(oe10, oe30, ie03, ie23);
    assertEquals4(oe11, oe31, ie04, ie24);
    assertEquals4(oe02, oe22, ie11, ie31);
    assertEquals4(oe03, oe23, ie12, ie32);
    assertEquals4(oe12, oe32, ie13, ie33);
    assertEquals4(oe13, oe33, ie14, ie34);
    Assert.assertTrue(ie00 == ie20);
    Assert.assertTrue(ie10 == ie30);
    SortNetMLP.SEdge oe40 = hiddenNode0.getOutputEdge(0);
    SortNetMLP.SEdge oe50 = hiddenNode1.getOutputEdge(0);
    SortNetMLP.SEdge oe60 = hiddenNode2.getOutputEdge(0);
    SortNetMLP.SEdge oe70 = hiddenNode3.getOutputEdge(0);
    SortNetMLP.SEdge oe41 = hiddenNode0.getOutputEdge(1);
    SortNetMLP.SEdge oe51 = hiddenNode1.getOutputEdge(1);
    SortNetMLP.SEdge oe61 = hiddenNode2.getOutputEdge(1);
    SortNetMLP.SEdge oe71 = hiddenNode3.getOutputEdge(1);
    SortNetMLP.SEdge ie40 = outputNode0.getInputEdge(0);
    SortNetMLP.SEdge ie50 = outputNode1.getInputEdge(0);
    SortNetMLP.SEdge ie41 = outputNode0.getInputEdge(1);
    SortNetMLP.SEdge ie51 = outputNode1.getInputEdge(1);
    SortNetMLP.SEdge ie42 = outputNode0.getInputEdge(2);
    SortNetMLP.SEdge ie52 = outputNode1.getInputEdge(2);
    SortNetMLP.SEdge ie43 = outputNode0.getInputEdge(3);
    SortNetMLP.SEdge ie53 = outputNode1.getInputEdge(3);
    SortNetMLP.SEdge ie44 = outputNode0.getInputEdge(4);
    SortNetMLP.SEdge ie54 = outputNode1.getInputEdge(4);
    assertEquals4(oe40, oe60, ie41, ie51);
    assertEquals4(oe41, oe61, ie42, ie52);
    assertEquals4(oe50, oe70, ie43, ie53);
    assertEquals4(oe51, oe71, ie44, ie54);
    Assert.assertTrue(ie40 == ie50);

    assertBetweenPairs(inputNode0, inputNode2, oe00, hiddenNode0, hiddenNode2);
    assertBetweenPairs(inputNode2, inputNode0, oe01, hiddenNode0, hiddenNode2);
    assertBetweenPairs(inputNode1, inputNode3, oe10, hiddenNode0, hiddenNode2);
    assertBetweenPairs(inputNode3, inputNode1, oe11, hiddenNode0, hiddenNode2);
    assertBiasEdge(ie00, hiddenNode0, hiddenNode2);

    assertBetweenPairs(inputNode0, inputNode2, oe02, hiddenNode1, hiddenNode3);
    assertBetweenPairs(inputNode2, inputNode0, oe03, hiddenNode1, hiddenNode3);
    assertBetweenPairs(inputNode1, inputNode3, oe12, hiddenNode1, hiddenNode3);
    assertBetweenPairs(inputNode3, inputNode1, oe13, hiddenNode1, hiddenNode3);
    assertBiasEdge(ie10, hiddenNode1, hiddenNode3);

    assertBetweenPairs(hiddenNode0, hiddenNode2, oe40, outputNode0, outputNode1);
    assertBetweenPairs(hiddenNode2, hiddenNode0, oe41, outputNode0, outputNode1);
    assertBetweenPairs(hiddenNode1, hiddenNode3, oe50, outputNode0, outputNode1);
    assertBetweenPairs(hiddenNode3, hiddenNode1, oe51, outputNode0, outputNode1);
    assertBiasEdge(ie40, outputNode0, outputNode1);
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
    SortNetMLP mlp = new SortNetMLP(1, NetworkShape.parseSetting("1,Sigmoid 1,ReLU"),
        new Optimizer.SGDFactory(), new Regularization.L1(), WeightInitializer.Type.normal.name());

    Assert.assertTrue(mlp.regularization instanceof Regularization.L1);

    // SortNetMLP always adds an output MNode with Activation.Sigmoid
    Assert.assertEquals(4, mlp.network.size());
    Assert.assertEquals(2, mlp.getLayer(3).size());
    SortNetMLP.SNode outputNode0 = mlp.getNode(3, 0);
    SortNetMLP.SNode outputNode1 = mlp.getNode(3, 1);
    Assert.assertTrue(outputNode0.getActivation() instanceof Activation.Sigmoid);
    Assert.assertTrue(outputNode1.getActivation() instanceof Activation.Sigmoid);
    Assert.assertEquals(0, outputNode0.getGroup());
    Assert.assertEquals(1, outputNode1.getGroup());

    Assert.assertEquals(2, mlp.getLayer(0).size());
    Assert.assertEquals(2, mlp.getLayer(1).size());
    Assert.assertEquals(2, mlp.getLayer(2).size());
    SortNetMLP.SNode inputNode0 = mlp.getNode(0, 0);
    SortNetMLP.SNode inputNode1 = mlp.getNode(0, 1);
    Assert.assertTrue(inputNode0.getActivation() instanceof Activation.Identity);
    Assert.assertTrue(inputNode1.getActivation() instanceof Activation.Identity);
    Assert.assertEquals(0, inputNode0.getGroup());
    Assert.assertEquals(1, inputNode1.getGroup());
    SortNetMLP.SNode hiddenNode0 = mlp.getNode(1, 0);
    SortNetMLP.SNode hiddenNode1 = mlp.getNode(1, 1);
    Assert.assertTrue(hiddenNode0.getActivation() instanceof Activation.Sigmoid);
    Assert.assertTrue(hiddenNode1.getActivation() instanceof Activation.Sigmoid);
    SortNetMLP.SNode hiddenNode2 = mlp.getNode(2, 0);
    SortNetMLP.SNode hiddenNode3 = mlp.getNode(2, 1);
    Assert.assertTrue(hiddenNode2.getActivation() instanceof Activation.ReLU);
    Assert.assertTrue(hiddenNode3.getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(0, inputNode0.getInputEdges().size());
    Assert.assertEquals(0, inputNode1.getInputEdges().size());
    Assert.assertEquals(2, inputNode0.getOutputEdges().size());
    Assert.assertEquals(2, inputNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode0.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode1.getInputEdges().size());
    Assert.assertEquals(2, hiddenNode0.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode1.getOutputEdges().size());
    Assert.assertEquals(3, hiddenNode2.getInputEdges().size());
    Assert.assertEquals(3, hiddenNode3.getInputEdges().size());
    Assert.assertEquals(2, hiddenNode2.getOutputEdges().size());
    Assert.assertEquals(2, hiddenNode3.getOutputEdges().size());
    Assert.assertEquals(3, outputNode0.getInputEdges().size());
    Assert.assertEquals(3, outputNode1.getInputEdges().size());
    Assert.assertEquals(0, outputNode0.getOutputEdges().size());
    Assert.assertEquals(0, outputNode1.getOutputEdges().size());

    SortNetMLP.SEdge outputEdge00 = inputNode0.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge10 = inputNode0.getOutputEdge(1);
    SortNetMLP.SEdge outputEdge01 = inputNode1.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge11 = inputNode1.getOutputEdge(1);
    SortNetMLP.SEdge inputEdge00 = hiddenNode0.getInputEdge(0);
    SortNetMLP.SEdge inputEdge01 = hiddenNode0.getInputEdge(1);
    SortNetMLP.SEdge inputEdge02 = hiddenNode0.getInputEdge(2);
    SortNetMLP.SEdge inputEdge10 = hiddenNode1.getInputEdge(0);
    SortNetMLP.SEdge inputEdge11 = hiddenNode1.getInputEdge(1);
    SortNetMLP.SEdge inputEdge12 = hiddenNode1.getInputEdge(2);
    assertEquals4(outputEdge00, outputEdge01, inputEdge01, inputEdge11);
    assertEquals4(outputEdge10, outputEdge11, inputEdge02, inputEdge12);
    Assert.assertTrue(inputEdge00 == inputEdge10);
    SortNetMLP.SEdge outputEdge20 = hiddenNode0.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge30 = hiddenNode0.getOutputEdge(1);
    SortNetMLP.SEdge outputEdge21 = hiddenNode1.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge31 = hiddenNode1.getOutputEdge(1);
    SortNetMLP.SEdge inputEdge20 = hiddenNode2.getInputEdge(0);
    SortNetMLP.SEdge inputEdge21 = hiddenNode2.getInputEdge(1);
    SortNetMLP.SEdge inputEdge22 = hiddenNode2.getInputEdge(2);
    SortNetMLP.SEdge inputEdge30 = hiddenNode3.getInputEdge(0);
    SortNetMLP.SEdge inputEdge31 = hiddenNode3.getInputEdge(1);
    SortNetMLP.SEdge inputEdge32 = hiddenNode3.getInputEdge(2);
    assertEquals4(outputEdge20, outputEdge21, inputEdge21, inputEdge31);
    assertEquals4(outputEdge30, outputEdge31, inputEdge22, inputEdge32);
    Assert.assertTrue(inputEdge20 == inputEdge30);
    SortNetMLP.SEdge outputEdge40 = hiddenNode2.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge50 = hiddenNode2.getOutputEdge(1);
    SortNetMLP.SEdge outputEdge41 = hiddenNode3.getOutputEdge(0);
    SortNetMLP.SEdge outputEdge51 = hiddenNode3.getOutputEdge(1);
    SortNetMLP.SEdge inputEdge40 = outputNode0.getInputEdge(0);
    SortNetMLP.SEdge inputEdge41 = outputNode0.getInputEdge(1);
    SortNetMLP.SEdge inputEdge42 = outputNode0.getInputEdge(2);
    SortNetMLP.SEdge inputEdge50 = outputNode1.getInputEdge(0);
    SortNetMLP.SEdge inputEdge51 = outputNode1.getInputEdge(1);
    SortNetMLP.SEdge inputEdge52 = outputNode1.getInputEdge(2);
    assertEquals4(outputEdge40, outputEdge41, inputEdge41, inputEdge51);
    assertEquals4(outputEdge50, outputEdge51, inputEdge42, inputEdge52);
    Assert.assertTrue(inputEdge40 == inputEdge50);

    assertBetweenPairs(inputNode0, inputNode1, outputEdge00, hiddenNode0, hiddenNode1);
    assertBetweenPairs(inputNode1, inputNode0, outputEdge10, hiddenNode0, hiddenNode1);
    assertBiasEdge(inputEdge00, hiddenNode0, hiddenNode1);

    assertBetweenPairs(hiddenNode0, hiddenNode1, outputEdge20, hiddenNode2, hiddenNode3);
    assertBetweenPairs(hiddenNode1, hiddenNode0, outputEdge30, hiddenNode2, hiddenNode3);
    assertBiasEdge(inputEdge20, hiddenNode2, hiddenNode3);

    assertBetweenPairs(hiddenNode2, hiddenNode3, outputEdge40, outputNode0, outputNode1);
    assertBetweenPairs(hiddenNode3, hiddenNode2, outputEdge50, outputNode0, outputNode1);
    assertBiasEdge(inputEdge40, outputNode0, outputNode1);
  }
}
