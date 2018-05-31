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

import static org.junit.Assert.*;
import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class RankBoostTest {
  private static final String MODEL_SRC = "{\n" +
      "  \"config\" : {\n" +
      "    \"algorithm\" : \"rankboost\",\n" +
      "    \"numIterations\" : 100,\n" +
      "    \"batchSize\" : 0,\n" +
      "    \"verbose\" : true,\n" +
      "    \"nomodel\" : false,\n" +
      "    \"params\" : {\n" +
      "      \"numSteps\" : 100,\n" +
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
      "      \"file\" : \"model/rankboost-model.json\"\n" +
      "    },\n" +
      "    \"evaluation\" : {\n" +
      "      \"evaluator\" : \"NDCG\",\n" +
      "      \"params\" : {\n" +
      "        \"k\" : 10\n" +
      "      }\n" +
      "    },\n" +
      "    \"report\" : {\n" +
      "      \"format\" : \"csv\",\n" +
      "      \"file\" : \"report/rankboost-report.csv\"\n" +
      "    }\n" +
      "  },\n" +
      "  \"features\" : [ 0, 1 ],\n" +
      "  \"thresholds\" : [ 1.1, 2.1 ],\n" +
      "  \"weights\" : [ 3.0, 4.0 ]\n" +
      "}";

  private static final String CONFIG_SRC = "{\n" +
      "  \"algorithm\" : \"rankboost\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"verbose\": true,\n" +
      "  \"params\" : {\n" +
      "    \"numSteps\" : 100,\n" +
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
      "    \"file\" : \"model/rankboost-model.json\"\n" +
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
      "    \"file\" : \"report/rankboost-report.csv\"\n" +
      "  }\n" +
      "}";
  private RankBoost rb;

  @Before
  public void setUp(){
    Reader reader = new StringReader(MODEL_SRC);
    rb = new RankBoost(reader);
  }

  @Test
  public void testReadWrite() throws Exception{
    ObjectMapper mapper = new ObjectMapper();
    Reader reader = new StringReader(CONFIG_SRC);
    RankBoost.RankBoostConfig config = mapper.readValue(reader, RankBoost.RankBoostConfig.class);

    Writer writer = new StringWriter();
    rb.writeModel(config, writer);
    Assert.assertEquals(MODEL_SRC, writer.toString());
  }

  @Test
  public void predict() {
    double[][] docs = {
        {1.0 , 2.0  },
        {1.1 , 2.0 },
        {1.1 , 2.1 },
        {1.0 , 2.1 },
        {0.0 , 0.0 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    int[] labels = {0, 1, 2, 1, 0};
    addLabels(docList, labels);

    Assert.assertEquals(0d, rb.predict(docList.get(0).getFeatures()), 0.01);
    Assert.assertEquals(3d, rb.predict(docList.get(1).getFeatures()), 0.01);
    Assert.assertEquals(7d, rb.predict(docList.get(2).getFeatures()), 0.01);
    Assert.assertEquals(4d, rb.predict(docList.get(3).getFeatures()), 0.01);
    Assert.assertEquals(0d, rb.predict(docList.get(4).getFeatures()), 0.01);

  }
}