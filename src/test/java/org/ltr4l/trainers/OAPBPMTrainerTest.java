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

public class OAPBPMTrainerTest {

  static final String JSON_SRC = "{\n" +
      "  \"algorithm\" : \"OAP\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"params\" : {\n" +
      "    \"bernoulli\" : 0.0642,\n" +
      "    \"N\" : 100\n" +
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
      "    \"file\" : \"oap-model.json\"\n" +
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
      "    \"file\" : \"oap-report.csv\"\n" +
      "  }\n" +
      "}\n";

  private OAPBPMTrainer.OAPBPMConfig config;

  @Before
  public void setUp() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    config = mapper.readValue(JSON_SRC, OAPBPMTrainer.getCC());
  }

  @Test
  public void test() throws Exception {
    Assert.assertEquals(100, config.getPNum());
    Assert.assertEquals(0.0642, config.getBernNum(), 0.000001);
  }
}
