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
import org.ltr4l.nn.AbstractEdge.AbstractFFEdge;
import org.ltr4l.nn.AbstractNode.Node;

public abstract class MLPTestBase<N extends Node, E extends AbstractFFEdge> {

  protected void assertBetweenNodes(N sn, int i, N dn, int j) throws Exception {
    //Because N is an extension of Node, the saved Edges should be of type AbstractFFEdge.
    AbstractFFEdge oe = (AbstractFFEdge) sn.getOutputEdge(i);
    AbstractFFEdge ie = (AbstractFFEdge) dn.getInputEdge(j);
    Assert.assertTrue(oe == ie);
    Assert.assertTrue(sn == oe.getSource());
    Assert.assertTrue(dn == oe.getDestination());
  }

  protected void assertBiasEdge(E edge, N dn) throws Exception {
    Assert.assertNull(edge.getSource());
    Assert.assertTrue(dn == edge.getDestination());
  }
}
