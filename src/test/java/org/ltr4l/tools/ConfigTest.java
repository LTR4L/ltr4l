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
import java.util.Properties;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;

public class ConfigTest {

  @Test
  public void testGetStrProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "strval");
    Assert.assertEquals("strval", Config.getStrProp(props, "prop1", "defval"));
    Assert.assertEquals("defval", Config.getStrProp(props, "prop2", "defval"));
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetReqStrProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "100");
    Config.getReqStrProp(props, "prop2");
  }

  @Test
  public void testGetIntProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "100");
    Assert.assertEquals(100, Config.getIntProp(props, "prop1", 200));
    Assert.assertEquals(200, Config.getIntProp(props, "prop2", 200));
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetReqIntProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "100");
    Config.getReqIntProp(props, "prop2");
  }

  @Test
  public void testGetDoubleProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "100");
    Assert.assertEquals(100, Config.getDoubleProp(props, "prop1", 200), 0.001);
    Assert.assertEquals(200, Config.getDoubleProp(props, "prop2", 200), 0.001);
  }

  @Test(expected = IllegalArgumentException.class)
  public void testGetReqDoubleProp() throws Exception {
    Properties props = new Properties();
    props.setProperty("prop1", "100");
    Config.getReqDoubleProp(props, "prop2");
  }

  @Test
  public void testDefaultValues() throws Exception {
    String strConfig = "name:OAP\n";
    Config config = Config.get(new StringReader(strConfig));

    Assert.assertEquals(100, config.getNumIterations());
    Assert.assertEquals(0, config.getLearningRate(), 0.001);
    Assert.assertEquals("zero", config.getWeightInit());
    Assert.assertEquals(0.03, config.getBernNum(), 0.0001);
    Assert.assertEquals(1, config.getPNum());
    Assert.assertEquals(0, config.getReguRate(), 0.0001);
  }

  @Test
  public void testSimple() throws Exception {
    String strConfig = "name:OAP\n" +
        "numIterations:200\n" +
        "bernoulli:0.03\n" +
        "N:250\n" +
        "reguRate:0.01\n";
    Config config = Config.get(new StringReader(strConfig));

    Assert.assertEquals("OAP", config.getName());
    Assert.assertEquals(200, config.getNumIterations());
    Assert.assertEquals(0.03, config.getBernNum(), 0.0001);
    Assert.assertEquals(250, config.getPNum());
    Assert.assertEquals(0.01, config.getReguRate(), 0.0001);
  }

  @Test
  public void testGetOptimizerFactory() throws Exception {
    Config config1 = Config.get(new StringReader("name:OAP\noptimizer:Adam"));
    Assert.assertTrue(config1.getOptFact() instanceof Optimizer.AdamFactory);

    Config config2 = Config.get(new StringReader("name:OAP\noptimizer:SGD"));
    Assert.assertTrue(config2.getOptFact() instanceof Optimizer.SGDFactory);

    Config config3 = Config.get(new StringReader("name:OAP\noptimizer:momentum"));
    Assert.assertTrue(config3.getOptFact() instanceof Optimizer.MomentumFactory);

    Config config4 = Config.get(new StringReader("name:OAP\noptimizer:nesterov"));
    Assert.assertTrue(config4.getOptFact() instanceof Optimizer.NesterovFactory);

    // if unknown optimizer is specified, it returns SGD
    Config config5 = Config.get(new StringReader("name:OAP\noptimizer:myGreatestOptimizer!"));
    Assert.assertTrue(config5.getOptFact() instanceof Optimizer.SGDFactory);
  }

  @Test
  public void testGetRegularizationFunction() throws Exception {
    Config config1 = Config.get(new StringReader("name:OAP\nreguFunction:L1"));
    Assert.assertTrue(config1.getReguFunction() instanceof Regularization.L1);

    Config config2 = Config.get(new StringReader("name:OAP\nreguFunction:L2"));
    Assert.assertTrue(config2.getReguFunction() instanceof Regularization.L2);

    // if unknown regularization function is specified, it returns L2
    Config config3 = Config.get(new StringReader("name:OAP\nreguFunction:myGreatestRegularizationFunc!"));
    Assert.assertTrue(config3.getReguFunction() instanceof Regularization.L2);
  }

  @Test
  public void testGetWeightInit() throws Exception {
    Config config1 = Config.get(new StringReader("name:OAP\nweightInit:normal"));
    Assert.assertEquals("normal", config1.getWeightInit());

    Config config2 = Config.get(new StringReader("name:OAP\nweightInit:xavier"));
    Assert.assertEquals("xavier", config2.getWeightInit());

    Config config3 = Config.get(new StringReader("name:OAP\nweightInit:zero"));
    Assert.assertEquals("zero", config3.getWeightInit());

    Config config4 = Config.get(new StringReader("name:OAP\nweightInit:somethingElse"));
    Assert.assertEquals("somethingElse", config4.getWeightInit());
  }

  @Test
  public void testGetNetworkShape() throws Exception {
    Config config1 = Config.get(new StringReader("name:OAP\nlayers:5,Identity 1,Sigmoid"));
    NetworkShape ns1 = config1.getNetworkShape();
    Assert.assertEquals(2, ns1.size());
    Assert.assertEquals(5, ns1.getLayerSetting(0).getNum());
    Assert.assertTrue(ns1.getLayerSetting(0).getActivation() instanceof Activation.Identity);
    Assert.assertEquals(1, ns1.getLayerSetting(1).getNum());
    Assert.assertTrue(ns1.getLayerSetting(1).getActivation() instanceof Activation.Sigmoid);

    Config config2 = Config.get(new StringReader("name:OAP\nlayers:15,Sigmoid"));
    NetworkShape ns2 = config2.getNetworkShape();
    Assert.assertEquals(1, ns2.size());
    Assert.assertEquals(15, ns2.getLayerSetting(0).getNum());
    Assert.assertTrue(ns2.getLayerSetting(0).getActivation() instanceof Activation.Sigmoid);
  }
}
