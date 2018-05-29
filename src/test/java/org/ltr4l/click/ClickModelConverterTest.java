package org.ltr4l.click;

import org.junit.Assert;
import org.junit.Test;

import java.io.*;
import java.nio.charset.StandardCharsets;

import static org.junit.Assert.*;

public class ClickModelConverterTest {
  private static final String SRC_JSON = "{\n" +
      " \"data\": [\n" +
      "   {\n" +
      "     \"query\": \"iPhone\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docA\", \"docC\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"iPhone\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docA\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"Android\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docD\", \"docC\", \"docD\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"Android\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docA\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"Android\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": []\n" +
      "   }\n" +
      " ]\n" +
      "}";


  private static final String TARGET_JSON = "{\n" +
      "  \"idField\" : \"url\",\n" +
      "  \"queries\" : [ {\n" +
      "    \"qid\" : 0,\n" +
      "    \"query\" : \"iPhone\",\n" +
      "    \"docs\" : [ \"docE\", \"docD\", \"docC\", \"docB\", \"docA\" ]\n" +
      "  }, {\n" +
      "    \"qid\" : 1,\n" +
      "    \"query\" : \"Android\",\n" +
      "    \"docs\" : [ \"docE\", \"docD\", \"docC\", \"docB\", \"docA\" ]\n" +
      "  } ]\n" +
      "}";

  @Test
  public void testGetCMQuery() throws Exception{
    InputStream inputStream = new ByteArrayInputStream(SRC_JSON.getBytes(StandardCharsets.UTF_8));
    ClickModelConverter cmc = new ClickModelConverter(inputStream);
    OutputStream os = cmc.getQuery();
    String json = os.toString();
    Assert.assertEquals(TARGET_JSON, json);
  }
}