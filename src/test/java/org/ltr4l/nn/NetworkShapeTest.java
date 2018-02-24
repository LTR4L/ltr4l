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

public class NetworkShapeTest {

  @Test
  public void testParseSetting1() throws Exception {
    NetworkShape ns1 = NetworkShape.parseSetting("3,Identity");
    Assert.assertEquals(1, ns1.size());
    Assert.assertEquals(3, ns1.getLayerSetting(0).getNum());
    Assert.assertTrue(ns1.getLayerSetting(0).getActivation() instanceof Activation.Identity);
  }

  @Test
  public void testParseSetting2() throws Exception {
    NetworkShape ns1 = NetworkShape.parseSetting("3,Identity 5,Sigmoid");
    Assert.assertEquals(2, ns1.size());
    Assert.assertEquals(3, ns1.getLayerSetting(0).getNum());
    Assert.assertTrue(ns1.getLayerSetting(0).getActivation() instanceof Activation.Identity);
    Assert.assertEquals(5, ns1.getLayerSetting(1).getNum());
    Assert.assertTrue(ns1.getLayerSetting(1).getActivation() instanceof Activation.Sigmoid);
  }

  @Test
  public void testParseSetting3() throws Exception {
    NetworkShape ns1 = NetworkShape.parseSetting("9,ReLU 4,Identity 10,Sigmoid");
    Assert.assertEquals(3, ns1.size());
    Assert.assertEquals(9, ns1.getLayerSetting(0).getNum());
    Assert.assertTrue(ns1.getLayerSetting(0).getActivation() instanceof Activation.ReLU);
    Assert.assertEquals(4, ns1.getLayerSetting(1).getNum());
    Assert.assertTrue(ns1.getLayerSetting(1).getActivation() instanceof Activation.Identity);
    Assert.assertEquals(10, ns1.getLayerSetting(2).getNum());
    Assert.assertTrue(ns1.getLayerSetting(2).getActivation() instanceof Activation.Sigmoid);
  }
}
