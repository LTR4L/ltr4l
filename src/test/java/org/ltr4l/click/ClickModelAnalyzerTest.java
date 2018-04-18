package org.ltr4l.click;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;

public class ClickModelAnalyzerTest {

  @Test
  public void testCalcImpressionLog() throws Exception {
    String testJson = "{\n" +
      " \"data\": [\n" +
      "   {\n" +
      "     \"query\": \"iPhone\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docA\", \"docC\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"iPhone\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docC\", \"docE\" ]\n" +
      "   },\n" +
      "   {\n" +
      "     \"query\": \"Android\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": []\n" +
      "   }\n" +
      " ]\n" +
      "}";

    InputStream inputStream = new ByteArrayInputStream(testJson.getBytes("utf-8"));

    List<ImpressionLog> impressionLogList = ClickModels.getInstance().parseImpressionLog(inputStream);
    ClickModelAnalyzer clickModelAnalyzer = new ClickModelAnalyzer();
    Map<String, Map<String, Float>> clickRates = clickModelAnalyzer.calcClickRate(impressionLogList);

    Assert.assertTrue("",clickRates.containsKey("iPhone"));
    Assert.assertTrue("",clickRates.containsKey("Android"));
    Assert.assertTrue("",clickRates.get("iPhone").containsKey("docA"));
    Assert.assertTrue("",clickRates.get("iPhone").containsKey("docB"));
    Assert.assertEquals(clickRates.get("iPhone").get("docA"), 0.5f, 0.0f);
    Assert.assertEquals(clickRates.get("iPhone").get("docB"), 0.0f, 0.0f);
    Assert.assertEquals(clickRates.get("iPhone").get("docC"), 1.0f, 0.0f);
  }
}
