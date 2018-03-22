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

package org.ltr4l.trainers;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.tools.Regularization;

public class MLPTrainerTest {

  static final String JSON_SRC = "{\n" +
      "  \"algorithm\" : \"NNRank\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"params\" : {\n" +
      "    \"learningRate\" : 0.00001,\n" +
      "    \"optimizer\" : \"adam\",\n" +
      "    \"weightInit\" : \"xavier\",\n" +
      "    \"regularization\" : {\n" +
      "      \"regularizer\" : \"L1\",\n" +
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
      "    \"file\" : \"nnrank-model.json\"\n" +
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
      "    \"file\" : \"nnrank-report.csv\"\n" +
      "  }\n" +
      "}\n";

  private MLPTrainer.MLPConfig config;

  @Before
  public void setUp() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    config = mapper.readValue(JSON_SRC, MLPTrainer.getCC());
  }

  @Test
  public void test() throws Exception {
    Assert.assertEquals(0.00001, config.getLearningRate(), 0.0000001);
    Assert.assertEquals(0.01, config.getReguRate(), 0.0001);
    Assert.assertTrue(config.getReguFunction() instanceof Regularization.L1);
    Assert.assertEquals("xavier", config.getWeightInit());
    Assert.assertTrue(config.getOptFact().getOptimizer() instanceof Optimizer.Adam);
  }
}
