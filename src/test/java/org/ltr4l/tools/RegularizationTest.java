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

package org.ltr4l.tools;

import org.junit.Assert;
import org.junit.Test;

public class RegularizationTest {

  @Test
  public void testFactory() throws Exception {
    Assert.assertTrue(Regularization.RegularizationFactory.getRegularization(Regularization.Type.L1)
        instanceof Regularization.L1);
    Assert.assertTrue(Regularization.RegularizationFactory.getRegularization(Regularization.Type.L2)
        instanceof Regularization.L2);
  }

  @Test
  public void testL1Output() throws Exception {
    Regularization regu = Regularization.RegularizationFactory.getRegularization(Regularization.Type.L1);

    Assert.assertEquals(10, regu.output(10), 0.001);
    Assert.assertEquals(100, regu.output(-100), 0.001);
  }

  @Test
  public void testL1Derivative() throws Exception {
    Regularization regu = Regularization.RegularizationFactory.getRegularization(Regularization.Type.L1);

    Assert.assertEquals(1, regu.derivative(10), 0.001);
    Assert.assertEquals(-1, regu.derivative(-100), 0.001);
    Assert.assertEquals(0, regu.derivative(0), 0.001);
  }

  @Test
  public void testL2Output() throws Exception {
    Regularization regu = Regularization.RegularizationFactory.getRegularization(Regularization.Type.L2);

    Assert.assertEquals(50, regu.output(10), 0.001);
    Assert.assertEquals(5000, regu.output(-100), 0.001);
  }

  @Test
  public void testL2Derivative() throws Exception {
    Regularization regu = Regularization.RegularizationFactory.getRegularization(Regularization.Type.L2);

    Assert.assertEquals(10, regu.derivative(10), 0.001);
    Assert.assertEquals(-100, regu.derivative(-100), 0.001);
    Assert.assertEquals(0, regu.derivative(0), 0.001);
  }
}
