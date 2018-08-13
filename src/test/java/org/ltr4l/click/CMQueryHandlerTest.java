package org.ltr4l.click;

import org.junit.Assert;
import org.junit.Test;

import java.io.*;
import java.nio.charset.StandardCharsets;

public class CMQueryHandlerTest {
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

  private static final String SRC_JSON_NON_ASCII_QUERY = "{\n" +
    " \"data\": [\n" +
    "   {\n" +
    "     \"query\": \"アイフォーン\",\n" +
    "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
    "     \"clicks\": [ \"docA\", \"docC\" ]\n" +
    "   },\n" +
    "   {\n" +
    "     \"query\": \"アイフォーン\",\n" +
    "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
    "     \"clicks\": [ \"docA\" ]\n" +
    "   },\n" +
    "   {\n" +
    "     \"query\": \"アンドロイド\",\n" +
    "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
    "     \"clicks\": [ \"docD\", \"docC\", \"docD\" ]\n" +
    "   },\n" +
    "   {\n" +
    "     \"query\": \"アンドロイド\",\n" +
    "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
    "     \"clicks\": [ \"docA\" ]\n" +
    "   },\n" +
    "   {\n" +
    "     \"query\": \"アンドロイド\",\n" +
    "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
    "     \"clicks\": []\n" +
    "   }\n" +
    " ]\n" +
    "}";


  private static final String TARGET_JSON = "{\n" +
      "  \"idField\" : \"id\",\n" +
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

  private static final String TARGET_JSON_2 = "{\n" +
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

  private static final String TARGET_JSON3 = "{\n" +
    "  \"idField\" : \"id\",\n" +
    "  \"queries\" : [ {\n" +
    "    \"qid\" : 0,\n" +
    "    \"query\" : \"アイフォーン\",\n" +
    "    \"docs\" : [ \"docE\", \"docD\", \"docC\", \"docB\", \"docA\" ]\n" +
    "  }, {\n" +
    "    \"qid\" : 1,\n" +
    "    \"query\" : \"アンドロイド\",\n" +
    "    \"docs\" : [ \"docE\", \"docD\", \"docC\", \"docB\", \"docA\" ]\n" +
    "  } ]\n" +
    "}";

  @Test
  public void testGetCMQuery() throws Exception{
    InputStream inputStream = new ByteArrayInputStream(SRC_JSON.getBytes(StandardCharsets.UTF_8));
    CMQueryHandler cmc = new CMQueryHandler(inputStream);
    OutputStream os = cmc.getQuery();
    String json = os.toString();
    Assert.assertEquals(TARGET_JSON, json);
  }

  @Test
  public void testGetCMQuery2() throws Exception{
    InputStream inputStream = new ByteArrayInputStream(SRC_JSON.getBytes(StandardCharsets.UTF_8));
    CMQueryHandler cmc = new CMQueryHandler(inputStream, "url");
    OutputStream os = cmc.getQuery();
    String json = os.toString();
    Assert.assertEquals(TARGET_JSON_2, json);
  }

  @Test
  public void testGetCMQuery3() throws Exception{
    InputStream inputStream = new ByteArrayInputStream(SRC_JSON_NON_ASCII_QUERY.getBytes(System.getProperty("file.encoding")));
    CMQueryHandler cmc = new CMQueryHandler(inputStream, "id");
    OutputStream os = cmc.getQuery();
    String json = os.toString();
    Assert.assertEquals(TARGET_JSON3, json);
  }

}