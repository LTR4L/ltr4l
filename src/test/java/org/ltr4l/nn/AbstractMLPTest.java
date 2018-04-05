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

import java.io.StringReader;
import java.io.StringWriter;
import java.util.List;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.tools.*;
import org.ltr4l.trainers.MLPTrainer;

public class AbstractMLPTest {

  static final String JSON_CONFIG = "{\n" +
      "  \"algorithm\" : \"MyMLP\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"params\" : {\n" +
      "    \"learningRate\" : 0.00001,\n" +
      "    \"optimizer\" : \"adam\",\n" +
      "    \"weightInit\" : \"xavier\",\n" +
      "    \"regularization\" : {\n" +
      "      \"regularizer\" : \"L2\",\n" +
      "      \"rate\" : 0.01\n" +
      "    },\n" +
      "    \"layers\" : [\n" +
      "      {\n" +
      "        \"activator\" : \"Sigmoid\",\n" +
      "        \"num\" : 3\n" +
      "      },\n" +
      "      {\n" +
      "        \"activator\" : \"Identity\",\n" +
      "        \"num\" : 2\n" +
      "      },\n" +
      "      {\n" +
      "        \"activator\" : \"ReLU\",\n" +
      "        \"num\" : 4\n" +
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
      "    \"file\" : \"mymlp-model.json\"\n" +
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
      "    \"file\" : \"mymlp-report.csv\"\n" +
      "  }\n" +
      "}\n";

  @Test
  public void testModelWriteRead() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    MLPTrainer.MLPConfig config = mapper.readValue(new StringReader(JSON_CONFIG), MLPTrainer.MLPConfig.class);

    MyMLP mlpW = new MyMLP(2, config.getNetworkShape(), new Optimizer.SGDFactory(),
        Regularization.RegularizationFactory.getRegularization(Regularization.DEFAULT), WeightInitializer.Type.sequence.name());

    StringWriter savedModel = new StringWriter();
    mlpW.writeModel(config, savedModel);

    List<List<MLP.MNode>> modelR = mlpW.readModel(new StringReader(savedModel.toString()));
    NetworkTestUtil ntu = new NetworkTestUtil<MLP.MNode>();

    Assert.assertEquals(4, modelR.size());
    ntu.assertLayer(modelR, 0, "|");    // inputLayer must have two nodes that don't have any inputEdges
    ntu.assertLayer(modelR, 1, "0,1,2|0,3,4|0,5,6");
    ntu.assertLayer(modelR, 2, "0,7,8,9|0,10,11,12");
    ntu.assertLayer(modelR, 3, "0,13,14|0,15,16|0,17,18|0,19,20");
  }

  static class MyMLP extends AbstractMLP<MLP.MNode, MLP.Edge> {

    public MyMLP(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
      super(inputDim, networkShape, optFact, regularization, weightModel);
    }

    @Override
    protected MLP.MNode constructNode(Activation activation) {
      return new MLP.MNode(activation);
    }

    @Override
    protected MLP.Edge constructEdge(MLP.MNode source, MLP.MNode destination, Optimizer opt, double weight) {
      return new MLP.Edge(source, destination, opt, weight);
    }

    @Override
    protected void addOutputs(NetworkShape networkShape) {

    }

    @Override
    public void backProp(org.ltr4l.tools.Error errorFunc, double... target) {

    }

    @Override
    public void updateWeights(double lrRate, double rgRate) {

    }
  }
}
