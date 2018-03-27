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

import java.io.StringReader;
import java.io.StringWriter;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.nn.PRank;
import org.ltr4l.tools.Config;

public class PRankTest {

  static final String JSON_CONFIG = "{\n" +
      "  \"algorithm\" : \"PRank\",\n" +
      "  \"numIterations\" : 100,\n" +
      "\n" +
      "  \"dataSet\" : {\n" +
      "    \"training\" : \"data/MQ2008/Fold1/train.txt\",\n" +
      "    \"validation\" : \"data/MQ2008/Fold1/vali.txt\",\n" +
      "    \"test\" : \"data/MQ2008/Fold1/test.txt\"\n" +
      "  },\n" +
      "\n" +
      "  \"model\" : {\n" +
      "    \"format\" : \"json\",\n" +
      "    \"file\" : \"prank-model.json\"\n" +
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
      "    \"file\" : \"prank-report.csv\"\n" +
      "  }\n" +
      "}\n";

  @Test
  public void testModelWriteRead() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    Config config = mapper.readValue(new StringReader(JSON_CONFIG), Config.class);

    PRank prankW = new PRank(6, 3);
    prankW.weights = new double[]{1.1, 2.2, 3.3, 4.4, 5.5, 6.6};
    prankW.thresholds = new double[]{-10.5, 0.123, 34.5};

    StringWriter savedModel = new StringWriter();
    prankW.writeModel(config, savedModel);

    PRank prankR = PRank.readModel(new StringReader(savedModel.toString()));

    Assert.assertEquals(6, prankR.weights.length);
    Assert.assertEquals(1.1, prankR.weights[0], 0.001);
    Assert.assertEquals(2.2, prankR.weights[1], 0.001);
    Assert.assertEquals(3.3, prankR.weights[2], 0.001);
    Assert.assertEquals(4.4, prankR.weights[3], 0.001);
    Assert.assertEquals(5.5, prankR.weights[4], 0.001);
    Assert.assertEquals(6.6, prankR.weights[5], 0.001);

    Assert.assertEquals(3, prankR.thresholds.length);
    Assert.assertEquals(-10.5, prankR.thresholds[0], 0.001);
    Assert.assertEquals(0.123, prankR.thresholds[1], 0.001);
    Assert.assertEquals(34.5, prankR.thresholds[2], 0.001);
  }
}
