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

import java.io.StringReader;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;

public class ConfigTest {

  @Test
  public void testSimple() throws Exception {
    String strConfig = "name:OAP\n" +
        "numIterations:100\n" +
        "bernoulli:0.03\n" +
        "N:250\n" +
        "reguRate:0.01\n";
    Config config = Config.get(new StringReader(strConfig));

    Assert.assertEquals("OAP", config.getName());
    Assert.assertEquals(100, config.getNumIterations());
    Assert.assertEquals(0.03, config.getBernNum(), 0.0001);
    Assert.assertEquals(250, config.getPNum());
    Assert.assertEquals(0.01, config.getReguRate(), 0.0001);
  }

  @Test
  public void testGetOptimizerFactory() throws Exception {
    Config config1 = Config.get(new StringReader("optimizer:Adam"));
    Assert.assertTrue(config1.getOptFact() instanceof Optimizer.AdamFactory);

    Config config2 = Config.get(new StringReader("optimizer:SGD"));
    Assert.assertTrue(config2.getOptFact() instanceof Optimizer.sgdFactory);

    Config config3 = Config.get(new StringReader("optimizer:momentum"));
    Assert.assertTrue(config3.getOptFact() instanceof Optimizer.MomentumFactory);

    Config config4 = Config.get(new StringReader("optimizer:nesterov"));
    Assert.assertTrue(config4.getOptFact() instanceof Optimizer.NesterovFactory);

    // if unknown optimizer is specified, it returns SGD
    Config config5 = Config.get(new StringReader("optimizer:myGreatestOptimizer!"));
    Assert.assertTrue(config5.getOptFact() instanceof Optimizer.sgdFactory);
  }

  @Test
  public void testGetRegularizationFunction() throws Exception {
    Config config1 = Config.get(new StringReader("reguFunction:L1"));
    Assert.assertTrue(config1.getReguFunction() instanceof Regularization.L1);

    Config config2 = Config.get(new StringReader("reguFunction:L2"));
    Assert.assertTrue(config2.getReguFunction() instanceof Regularization.L2);

    // if unknown regularization function is specified, it returns L2
    Config config3 = Config.get(new StringReader("reguFunction:myGreatestRegularizationFunc!"));
    Assert.assertTrue(config3.getReguFunction() instanceof Regularization.L2);
  }

  @Test
  public void testGetWeightInit() throws Exception {
    Config config1 = Config.get(new StringReader("weightInit:normal"));
    Assert.assertEquals("normal", config1.getWeightInit());

    Config config2 = Config.get(new StringReader("weightInit:xavier"));
    Assert.assertEquals("xavier", config2.getWeightInit());

    Config config3 = Config.get(new StringReader("weightInit:zero"));
    Assert.assertEquals("zero", config3.getWeightInit());

    Config config4 = Config.get(new StringReader("weightInit:somethingElse"));
    Assert.assertEquals("somethingElse", config4.getWeightInit());
  }

  @Test
  public void testGetNetworkShape() throws Exception {
    Config config1 = Config.get(new StringReader("layers:5,Identity 1,Sigmoid"));
    Object[][] obj1 = config1.getNetworkShape();
    Assert.assertEquals(2, obj1.length);
    Assert.assertEquals(2, obj1[0].length);
    Assert.assertEquals(5, obj1[0][0]);
    Assert.assertTrue(obj1[0][1] instanceof Activation.Identity);
    Assert.assertEquals(2, obj1[1].length);
    Assert.assertEquals(1, obj1[1][0]);
    Assert.assertTrue(obj1[1][1] instanceof Activation.Sigmoid);

    Config config2 = Config.get(new StringReader("layers:15,Sigmoid"));
    Object[][] obj2 = config2.getNetworkShape();
    Assert.assertEquals(1, obj2.length);
    Assert.assertEquals(2, obj2[0].length);
    Assert.assertEquals(15, obj2[0][0]);
    Assert.assertTrue(obj2[0][1] instanceof Activation.Sigmoid);
  }
}
