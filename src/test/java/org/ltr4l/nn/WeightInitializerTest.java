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

public class WeightInitializerTest {

  @Test
  public void testGet() throws Exception {
    Assert.assertTrue(WeightInitializer.get(WeightInitializer.Type.xavier.name(), 10) instanceof WeightInitializer);
    Assert.assertTrue(WeightInitializer.get(WeightInitializer.Type.normal.name(), 10) instanceof WeightInitializer);
    Assert.assertTrue(WeightInitializer.get(WeightInitializer.Type.uniform.name(), 10) instanceof WeightInitializer);
    Assert.assertTrue(WeightInitializer.get(WeightInitializer.Type.zero.name(), 10) instanceof WeightInitializer);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetUnknownType() throws Exception {
    WeightInitializer.get("MyGreatestWeightInitializer!", 100);
  }

  @Test
  public void testGetNumWeights() throws Exception {
    Assert.assertEquals(40, WeightInitializer.getNumWeights(10, NetworkShape.parseSetting("4,Identity")));
    Assert.assertEquals(32, WeightInitializer.getNumWeights(3, NetworkShape.parseSetting("4,Identity 5,Sigmoid")));
    Assert.assertEquals(82, WeightInitializer.getNumWeights(3, NetworkShape.parseSetting("4,Identity 5,Sigmoid 10,ReLU")));
  }

  @Test
  public void testGetInitialBias() throws Exception {
    WeightInitializer weightInit1 = WeightInitializer.get(WeightInitializer.Type.zero.name(), 100);
    Assert.assertEquals(0, weightInit1.getInitialBias(), 0.001);

    WeightInitializer weightInit2 = WeightInitializer.get(WeightInitializer.DEFAULT.name(), 100);
    Assert.assertEquals(0.01, weightInit2.getInitialBias(), 0.001);
  }

  @Test
  public void testSequenceWeight() throws Exception {
    WeightInitializer weightInit1 = WeightInitializer.get(WeightInitializer.Type.sequence.name(), 100);
    Assert.assertEquals(0, weightInit1.getInitialBias(), 0.001);
    Assert.assertEquals(1, weightInit1.getNextRandomInitialWeight(), 0.001);
    Assert.assertEquals(2, weightInit1.getNextRandomInitialWeight(), 0.001);
    Assert.assertEquals(3, weightInit1.getInitialBias(), 0.001);
    Assert.assertEquals(4, weightInit1.getNextRandomInitialWeight(), 0.001);
    Assert.assertEquals(5, weightInit1.getNextRandomInitialWeight(), 0.001);
  }
}
