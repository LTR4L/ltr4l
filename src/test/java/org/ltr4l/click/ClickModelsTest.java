package org.ltr4l.click;

import java.io.ByteArrayInputStream;
import java.io.File;
import java.io.InputStream;
import java.io.StringBufferInputStream;
import java.util.Arrays;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;

public class ClickModelsTest {

  @Test
  public void testParseImpressionLog() throws Exception {
    String testJson = "{\n" +
      " \"data\": [\n" +
      "   {\n" +
      "     \"query\": \"iPhone\",\n" +
      "     \"impressions\": [ \"docA\", \"docB\", \"docC\", \"docD\", \"docE\" ],\n" +
      "     \"clicks\": [ \"docA\", \"docC\" ]\n" +
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
    Assert.assertEquals("",impressionLogList.get(0).getQuery(), "iPhone");
    Assert.assertEquals("",impressionLogList.get(1).getQuery(), "Android");
    Assert.assertEquals("",impressionLogList.get(0).getImpressions(), Arrays.asList(new String[]{ "docA", "docB", "docC", "docD", "docE"}));
    Assert.assertEquals("",impressionLogList.get(1).getImpressions(), Arrays.asList(new String[]{ "docA", "docB", "docC", "docD", "docE"}));
    Assert.assertEquals("",impressionLogList.get(0).getClicks(), Arrays.asList(new String[]{ "docA", "docC"}));
    Assert.assertEquals("",impressionLogList.get(1).getClicks(), Arrays.asList(new String[]{ }));
  }

  @Test
  public void testParseImpressionLogFile() throws Exception {
    File file = new File("clickmodel/test.json");
    List<ImpressionLog> impressionLogList = ClickModels.getInstance().parseImpressionLog(file);
    Assert.assertEquals("",impressionLogList.get(0).getQuery(), "iPhone");
    Assert.assertEquals("",impressionLogList.get(4).getQuery(), "Android");
    Assert.assertEquals("",impressionLogList.get(0).getImpressions(), Arrays.asList(new String[]{ "docA", "docB", "docC", "docD", "docE"}));
    Assert.assertEquals("",impressionLogList.get(4).getImpressions(), Arrays.asList(new String[]{ "docA", "docB", "docC", "docD", "docE"}));
    Assert.assertEquals("",impressionLogList.get(0).getClicks(), Arrays.asList(new String[]{ "docA", "docC"}));
    Assert.assertEquals("",impressionLogList.get(4).getClicks(), Arrays.asList(new String[]{ }));
  }
}
