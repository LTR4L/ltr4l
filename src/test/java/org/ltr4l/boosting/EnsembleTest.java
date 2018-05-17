package org.ltr4l.boosting;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;

import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.List;

public class EnsembleTest {
  private static final String MODEL_SRC = "{\n" +
      "  \"config\" : {\n" +
      "    \"algorithm\" : \"LambdaMart\",\n" +
      "    \"numIterations\" : 100,\n" +
      "    \"batchSize\" : 0,\n" +
      "    \"verbose\" : true,\n" +
      "    \"nomodel\" : false,\n" +
      "    \"params\" : {\n" +
      "      \"numTrees\" : 15,\n" +
      "      \"numLeaves\" : 3,\n" +
      "      \"learningRate\" : 0.05,\n" +
      "      \"optimizer\" : \"adam\",\n" +
      "      \"weightInit\" : \"xavier\",\n" +
      "      \"regularization\" : {\n" +
      "        \"regularizer\" : \"L2\",\n" +
      "        \"rate\" : 0.01\n" +
      "      }\n" +
      "    },\n" +
      "    \"dataSet\" : {\n" +
      "      \"training\" : \"data/MQ2008/Fold1/train.txt\",\n" +
      "      \"validation\" : \"data/MQ2008/Fold1/vali.txt\",\n" +
      "      \"test\" : \"data/MQ2008/Fold1/test.txt\"\n" +
      "    },\n" +
      "    \"model\" : {\n" +
      "      \"format\" : \"json\",\n" +
      "      \"file\" : \"model/lambdamart-model.json\"\n" +
      "    },\n" +
      "    \"evaluation\" : {\n" +
      "      \"evaluator\" : \"NDCG\",\n" +
      "      \"params\" : {\n" +
      "        \"k\" : 10\n" +
      "      }\n" +
      "    },\n" +
      "    \"report\" : {\n" +
      "      \"format\" : \"csv\",\n" +
      "      \"file\" : \"report/lambdamart-report.csv\"\n" +
      "    }\n" +
      "  },\n" +
      "  \"treeModels\" : [ {\n" +
      "    \"config\" : null,\n" +
      "    \"leafIds\" : [ 0, 1, 3, 4, 2 ],\n" +
      "    \"featureIds\" : [ 0, 1, -1, -1, -1 ],\n" +
      "    \"thresh\" : [ 0.748092, 0.523628, \"-Infinity\", \"-Infinity\", \"-Infinity\" ],\n" +
      "    \"scores\" : [ 0.0, 0.0, 0.04655721205243154, -0.026720292627365028, 0.045137089557086 ]\n" +
      "  }, {\n" +
      "    \"config\" : null,\n" +
      "    \"leafIds\" : [ 0, 1, 3, 4, 2 ],\n" +
      "    \"featureIds\" : [ 1, 2, -1, -1, -1 ],\n" +
      "    \"thresh\" : [ 0.670009, 0.6, \"-Infinity\", \"-Infinity\", \"-Infinity\" ],\n" +
      "    \"scores\" : [ 0.0, 0.0, 0.09396994415691966, -0.1645747837427643, 0.10419659196565713 ]\n" +
      "  }, {\n" +
      "    \"config\" : null,\n" +
      "    \"leafIds\" : [ 0, 1, 2, 5, 6 ],\n" +
      "    \"featureIds\" : [ 2, -1, 0, -1, -1 ],\n" +
      "    \"thresh\" : [ 0.44423, \"-Infinity\", 0.7, \"-Infinity\", \"-Infinity\" ],\n" +
      "    \"scores\" : [ 0.0, 0.05492607939288196, 0.0, 0.16440961119308373, 0.06827215571207843 ]\n" +
      "  } ]\n" +
      "}";

  private static final String CONFIG_SRC = "{\n" +
      "  \"algorithm\" : \"LambdaMart\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"verbose\": true,\n" +
      "  \"params\" : {\n" +
      "    \"numTrees\" : 15,\n" +
      "    \"numLeaves\" : 3,\n" +
      "    \"learningRate\" : 0.05,\n" +
      "    \"optimizer\" : \"adam\",\n" +
      "    \"weightInit\" : \"xavier\",\n" +
      "    \"regularization\" : {\n" +
      "      \"regularizer\" : \"L2\",\n" +
      "      \"rate\" : 0.01\n" +
      "    }\n" +
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
      "    \"file\" : \"model/lambdamart-model.json\"\n" +
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
      "    \"file\" : \"report/lambdamart-report.csv\"\n" +
      "  }\n" +
      "}";


  private Ensemble ensemble;
  private Ensemble.TreeConfig config;

  @Before
  public void setUp() throws Exception{
    Reader reader = new StringReader(MODEL_SRC);
    ensemble = new Ensemble(reader);

    ObjectMapper mapper = new ObjectMapper();
    reader = new StringReader(CONFIG_SRC);
    config = mapper.readValue(reader, Ensemble.TreeConfig.class);
  }

  @Test
  public void testReadConstruction() throws Exception{ //testPredict tests behavior of construction.
    Assert.assertEquals(ensemble.getTrees().size(), 3);
  }

  @Test
  public void testWriteModel() throws Exception{
    Writer writer = new StringWriter();
    ensemble.writeModel(config, writer);
    String output = writer.toString();
    Assert.assertTrue(output.equals(MODEL_SRC));
  }

  @Test
  public void predict() throws Exception{ //TODO: Make more general
    double[][] samples = {
        {0.6, 0.5, 0.5},
        {0.7, 0.6, 0.6},
        {0.75, 0.7, 0.44}
    };
    List<Document> docs = TreeToolsTest.makeDocsWithFeatures(samples);

    double[] predictions = getPredictions(docs.get(0));
    Assert.assertEquals(predictions[0], 0.04656, 0.00001);
    Assert.assertEquals(predictions[1], 0.09397, 0.00001);
    Assert.assertEquals(predictions[2], 0.16441, 0.00001);
    Assert.assertEquals(ensemble.predict(docs.get(0).getFeatures()), predictions[3], 0.00001);

    predictions = getPredictions(docs.get(1));
    Assert.assertEquals(predictions[0], -0.02672, 0.00001);
    Assert.assertEquals(predictions[1], -0.16457, 0.00001);
    Assert.assertEquals(predictions[2], 0.06827, 0.00001);
    Assert.assertEquals(ensemble.predict(docs.get(1).getFeatures()), predictions[3], 0.00001);

    predictions = getPredictions(docs.get(2));
    Assert.assertEquals(predictions[0], 0.04514, 0.00001);
    Assert.assertEquals(predictions[1], 0.10420, 0.00001);
    Assert.assertEquals(predictions[2], 0.05493, 0.00001);
    Assert.assertEquals(ensemble.predict(docs.get(2).getFeatures()), predictions[3], 0.00001);
  }

  public double[] getPredictions(Document doc){
    double[] predictions = new double[ensemble.getTrees().size() + 1];
    double total = 0;
    for(int i = 0; i < ensemble.getTrees().size(); i++){
      double prediction = ensemble.getTree(i).predict(doc.getFeatures());
      total += prediction;
      predictions[i] = prediction;
    }
    predictions[predictions.length - 1] = total;
    return predictions;
  }
}