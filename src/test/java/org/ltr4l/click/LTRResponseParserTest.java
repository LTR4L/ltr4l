package org.ltr4l.click;

import org.junit.Assert;
import org.junit.Test;

import java.io.Reader;
import java.io.StringReader;
import java.util.Map;

public class LTRResponseHandlerTest {
  private static final String testJson = "{\n" +
      "    \"responseHeader\": {\n" +
      "        \"QTime\": 6, \n" +
      "        \"status\": 0\n" +
      "    }, \n" +
      "    \"results\": {\n" +
      "        \"command\": \"download\", \n" +
      "        \"procId\": 1476600502540, \n" +
      "        \"result\": {\n" +
      "            \"data\": {\n" +
      "                \"feature\": [\n" +
      "                    \"TF in name\", \n" +
      "                    \"TF in features\", \n" +
      "                    \"IDF in name\", \n" +
      "                    \"IDF in features\"\n" +
      "                ], \n" +
      "                \"queries\": [\n" +
      "                    {\n" +
      "                        \"docs\": [\n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    3.496508\n" +
      "                                ], \n" +
      "                                \"id\": \"F8V7067-APL-KIT\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    2, \n" +
      "                                    1, \n" +
      "                                    2.397895, \n" +
      "                                    3.496508\n" +
      "                                ], \n" +
      "                                \"id\": \"IW-02\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    3.496508\n" +
      "                                ], \n" +
      "                                \"id\": \"MA147LL/A\"\n" +
      "                            }\n" +
      "                        ], \n" +
      "                        \"qid\": 2, \n" +
      "                        \"query\": \"ipod\"\n" +
      "                    }, \n" +
      "                    {\n" +
      "                        \"docs\": [\n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"TWINX2048-3200PRO\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"VS1GB400C3\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"VDBDB1A16\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    0, \n" +
      "                                    3, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"0579B002\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"feature\": [\n" +
      "                                    0, \n" +
      "                                    1, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"EN7800GTX/2DHTV/256M\"\n" +
      "                            }\n" +
      "                        ], \n" +
      "                        \"qid\": 3, \n" +
      "                        \"query\": \"memory\"\n" +
      "                    }\n" +
      "                ]\n" +
      "            }\n" +
      "        }\n" +
      "    }\n" +
      "}";

  @Test
  public void getQueryMap() throws Exception{
    Reader reader = new StringReader(testJson);
    LTRResponseHandler parser = new LTRResponseHandler(reader);
    Map<String, LTRResponse.Doc[]> qMap = parser.getQueryMap();
    LTRResponse.Doc[] docs = qMap.get("ipod");
    Assert.assertEquals(docs.length, 3);

    LTRResponse.Doc doc = docs[0];
    Assert.assertEquals(doc.id, "F8V7067-APL-KIT");
    Assert.assertEquals(doc.feature[0], 1, 0.01);
    Assert.assertEquals(doc.feature[1], 0, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 3.496508, 0.01);

    doc = docs[1];
    Assert.assertEquals(doc.id, "IW-02");
    Assert.assertEquals(doc.feature[0], 2, 0.01);
    Assert.assertEquals(doc.feature[1], 1, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 3.496508, 0.01);

    doc = docs[2];
    Assert.assertEquals(doc.feature[0], 1, 0.01);
    Assert.assertEquals(doc.feature[1], 0, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 3.496508, 0.01);

    docs = qMap.get("memory");
    Assert.assertEquals(docs.length, 5);

    doc = docs[0];
    Assert.assertEquals(doc.id, "TWINX2048-3200PRO");
    Assert.assertEquals(doc.feature[0], 1, 0.01);
    Assert.assertEquals(doc.feature[1], 0, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 2.80336, 0.01);

    doc = docs[1];
    Assert.assertEquals(doc.id, "VS1GB400C3");
    Assert.assertEquals(doc.feature[0], 1, 0.01);
    Assert.assertEquals(doc.feature[1], 0, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 2.80336, 0.01);

    doc = docs[2];
    Assert.assertEquals(doc.id, "VDBDB1A16");
    Assert.assertEquals(doc.feature[0], 1, 0.01);
    Assert.assertEquals(doc.feature[1], 0, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 2.80336, 0.01);

    doc = docs[3];
    Assert.assertEquals(doc.id, "0579B002");
    Assert.assertEquals(doc.feature[0], 0, 0.01);
    Assert.assertEquals(doc.feature[1], 3, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 2.80336, 0.01);

    doc = docs[4];
    Assert.assertEquals(doc.id, "EN7800GTX/2DHTV/256M");
    Assert.assertEquals(doc.feature[0], 0, 0.01);
    Assert.assertEquals(doc.feature[1], 1, 0.01);
    Assert.assertEquals(doc.feature[2], 2.397895, 0.01);
    Assert.assertEquals(doc.feature[3], 2.80336, 0.01);
  }
}