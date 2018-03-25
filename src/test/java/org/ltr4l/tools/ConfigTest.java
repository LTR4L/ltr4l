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

import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.cli.CommandLine;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.cli.Train;

public class ConfigTest {

  static final String JSON_SRC = "{\n" +
      "  \"algorithm\" : \"FRankNet\",\n" +
      "  \"numIterations\" : 150,\n" +
      "  \"params\" : {\n" +
      "    \"learningRate\" : 0.001,\n" +
      "    \"optimizer\" : \"sgd\",\n" +
      "    \"weightInit\" : \"normal\",\n" +
      "    \"regularization\" : {\n" +
      "      \"regularizer\" : \"L2\",\n" +
      "      \"rate\" : 0.01\n" +
      "    },\n" +
      "    \"layers\" : [\n" +
      "      {\n" +
      "        \"activator\" : \"Identity\",\n" +
      "        \"num\" : 5\n" +
      "      },\n" +
      "      {\n" +
      "        \"activator\" : \"Sigmoid\",\n" +
      "        \"num\" : 1\n" +
      "      }\n" +
      "    ]\n" +
      "  },\n" +
      "\n" +
      "  \"dataSet\" : {\n" +
      "    \"training\" : \"data/MQ2008/Fold1/train.txt\",\n" +
      "    \"validation\" : \"data/MQ2008/Fold1/vali.txt\",\n" +
      "    \"test\" : \"data/MQ2008/Fold1/test.txt\"\n" +
      "  },\n" +
      "\n" +
      "  \"model\" : {\n" +
      "    \"format\" : \"json\",\n" +
      "    \"file\" : \"franknet-model.json\"\n" +
      "  },\n" +
      "\n" +
      "  \"evaluation\" : {\n" +
      "    \"evaluator\" : \"NDCG\",\n" +
      "    \"params\" : {\n" +
      "      \"k\" : 10\n" +
      "    }\n" +
      "  },\n" +
      "\n" +
      "  \"report\" : {\n" +
      "    \"format\" : \"csv\",\n" +
      "    \"file\" : \"franknet-report.csv\"\n" +
      "  }\n" +
      "}\n";

  private Config config;

  @Before
  public void setUp() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    config = mapper.readValue(JSON_SRC, Config.class);
  }

  @Test
  public void test() throws Exception {
    Assert.assertEquals("FRankNet", config.algorithm);
    Assert.assertEquals(150, config.numIterations);
    Assert.assertEquals("data/MQ2008/Fold1/train.txt", config.dataSet.training);
    Assert.assertEquals("data/MQ2008/Fold1/vali.txt", config.dataSet.validation);
    Assert.assertEquals("data/MQ2008/Fold1/test.txt", config.dataSet.test);
    Assert.assertEquals("json", config.model.format);
    Assert.assertEquals("franknet-model.json", config.model.file);
    Assert.assertEquals("NDCG", config.evaluation.evaluator);
    Assert.assertEquals(10, Config.getReqInt(config.evaluation.params, "k"));
    Assert.assertEquals("csv", config.report.format);
    Assert.assertEquals("franknet-report.csv", config.report.file);

    Assert.assertEquals(0.001, Config.getReqDouble(config.params, "learningRate"), 0.00001);
    Assert.assertEquals("sgd", Config.getReqString(config.params, "optimizer"));
    Assert.assertEquals("normal", Config.getReqString(config.params, "weightInit"));
    Assert.assertEquals("L2", Config.getString(Config.getReqParams(config.params, "regularization"), "regularizer", "L1"));
    Assert.assertEquals(0.01, Config.getReqDouble(Config.getReqParams(config.params, "regularization"), "rate"), 0.0001);

    List<Map<String, Object>> layersParams = Config.getReqArrayParams(config.params, "layers");
    Assert.assertEquals(2, layersParams.size());
    Assert.assertEquals("Identity", Config.getReqString(layersParams.get(0), "activator"));
    Assert.assertEquals(5, Config.getReqInt(layersParams.get(0), "num"));
    Assert.assertEquals("Sigmoid", Config.getReqString(layersParams.get(1), "activator"));
    Assert.assertEquals(1, Config.getReqInt(layersParams.get(1), "num"));
  }

  @Test
  public void testReq(){
    try {
      Config.getReqString(config.params, "optimizer1");
      Assert.fail("NPE must occur!");
    }
    catch (NullPointerException expected){}
    try {
      Config.getReqInt(config.params, "num1");
      Assert.fail("NPE must occur!");
    }
    catch (NullPointerException expected){}
    try {
      Config.getReqDouble(config.params, "learningRate1");
      Assert.fail("NPE must occur!");
    }
    catch (NullPointerException expected){}
  }

  @Test
  public void testOverrideByNull() throws Exception {
    config.overrideBy(null);
    test();
  }

  @Test
  public void testOverride() throws Exception {
    CommandLine line = Train.getCommandLine(Train.createOptions(),
        new String[] {"franknet",
            "-iterations", "500",
            "-training", "mytrain.txt",
            "-validation", "myvali.txt",
            "-model", "mymodel.json",
            "-report", "myreport.csv"});
    String configPath = Train.getConfigPath(line, line.getArgs());
    Config override = Train.createOptionalConfig(configPath, line);
    config.overrideBy(override);

    Assert.assertEquals("FRankNet", config.algorithm);
    Assert.assertEquals(500, config.numIterations);
    Assert.assertEquals("mytrain.txt", config.dataSet.training);
    Assert.assertEquals("myvali.txt", config.dataSet.validation);
    Assert.assertEquals("data/MQ2008/Fold1/test.txt", config.dataSet.test);
    Assert.assertEquals("json", config.model.format);
    Assert.assertEquals("mymodel.json", config.model.file);
    Assert.assertEquals("NDCG", config.evaluation.evaluator);
    Assert.assertEquals(10, Config.getReqInt(config.evaluation.params, "k"));
    Assert.assertEquals("csv", config.report.format);
    Assert.assertEquals("myreport.csv", config.report.file);

    Assert.assertEquals(0.001, Config.getReqDouble(config.params, "learningRate"), 0.00001);
    Assert.assertEquals("momentum", Config.getReqString(config.params, "optimizer"));
    Assert.assertEquals("normal", Config.getReqString(config.params, "weightInit"));
    Assert.assertEquals("L2", Config.getString(Config.getReqParams(config.params, "regularization"), "regularizer", "L1"));
    Assert.assertEquals(0.01, Config.getReqDouble(Config.getReqParams(config.params, "regularization"), "rate"), 0.0001);

    List<Map<String, Object>> layersParams = Config.getReqArrayParams(config.params, "layers");
    Assert.assertEquals(1, layersParams.size());
    Assert.assertEquals("Sigmoid", Config.getReqString(layersParams.get(0), "activator"));
    Assert.assertEquals(5, Config.getReqInt(layersParams.get(0), "num"));
  }
}
