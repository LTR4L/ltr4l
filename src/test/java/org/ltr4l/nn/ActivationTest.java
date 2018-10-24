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

public class ActivationTest {

  @Test
  public void testIdentity() throws Exception {
    Activation act = Activation.Type.Identity;

    Assert.assertEquals(10, act.output(10), 0.001);
    Assert.assertEquals(-100, act.output(-100), 0.001);
    Assert.assertEquals(0, act.output(0), 0.001);

    Assert.assertEquals(1, act.derivative(10), 0.001);
    Assert.assertEquals(1, act.derivative(-100), 0.001);
    Assert.assertEquals(1, act.derivative(0), 0.001);
  }

  @Test
  public void testSigmoid() throws Exception {
    Activation act = Activation.Type.Sigmoid;

    Assert.assertEquals(0.731059, act.output(1), 0.001);
    Assert.assertEquals(0.5, act.output(0), 0.001);
    Assert.assertEquals(0.268941, act.output(-1), 0.001);
  }

  @Test
  public void testReLU() throws Exception {
    Activation act = Activation.Type.ReLU;

    Assert.assertEquals(10, act.output(10), 0.001);
    Assert.assertEquals(0.01, act.output(0), 0.001);
    Assert.assertEquals(0.01, act.output(-100), 0.001);
    Assert.assertEquals(1, act.derivative(10), 0.001);
    Assert.assertEquals(0, act.derivative(0), 0.001);
    Assert.assertEquals(0, act.derivative(-100), 0.001);
  }

  @Test
  public void testLeakyReLU() throws Exception {
    Activation act = Activation.Type.LeakyReLU;

    Assert.assertEquals(10, act.output(10), 0.001);
    Assert.assertEquals(0d, act.output(0), 0.001);
    Assert.assertEquals(-1d, act.output(-100), 0.001);
    Assert.assertEquals(1, act.derivative(10), 0.001);
    Assert.assertEquals(0.01, act.derivative(0), 0.001);
    Assert.assertEquals(0.01, act.derivative(-100), 0.001);
  }

  @Test
  public void testTanH() throws Exception {
    Activation act = Activation.Type.TanH;

    Assert.assertEquals(0.7616, act.output(1), 0.001);
    Assert.assertEquals(0d, act.output(0), 0.001);
    Assert.assertEquals(-0.7616, act.output(-1), 0.001);
  }
}
