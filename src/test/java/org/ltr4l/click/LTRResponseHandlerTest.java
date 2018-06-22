package org.ltr4l.click;

import org.junit.Assert;
import org.junit.Test;

import java.io.ByteArrayInputStream;
import java.io.InputStream;
import java.io.Reader;
import java.io.StringReader;
import java.nio.charset.StandardCharsets;
import java.util.Map;

public class LTRResponseHandlerTest {
  private static final String testResponse = "{\n" +
      "    \"responseHeader\": {\n" +
      "        \"QTime\": 6, \n" +
      "        \"status\": 0\n" +
      "    }, \n" +
      "    \"results\": {\n" +
      "        \"command\": \"download\", \n" +
      "        \"procId\": 1476600502540, \n" +
      "        \"result\": {\n" +
      "            \"data\": {\n" +
      "                \"lucene\": [\n" +
      "                    \"TF in name\", \n" +
      "                    \"TF in features\", \n" +
      "                    \"IDF in name\", \n" +
      "                    \"IDF in features\"\n" +
      "                ], \n" +
      "                \"queries\": [\n" +
      "                    {\n" +
      "                        \"docs\": [\n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    3.496508\n" +
      "                                ], \n" +
      "                                \"id\": \"F8V7067-APL-KIT\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
      "                                    2, \n" +
      "                                    1, \n" +
      "                                    2.397895, \n" +
      "                                    3.496508\n" +
      "                                ], \n" +
      "                                \"id\": \"IW-02\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
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
      "                                \"lucene\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"TWINX2048-3200PRO\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"VS1GB400C3\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
      "                                    1, \n" +
      "                                    0, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"VDBDB1A16\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
      "                                    0, \n" +
      "                                    3, \n" +
      "                                    2.397895, \n" +
      "                                    2.80336\n" +
      "                                ], \n" +
      "                                \"id\": \"0579B002\"\n" +
      "                            }, \n" +
      "                            {\n" +
      "                                \"lucene\": [\n" +
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

  private static final String RESPONSE_JSON = "{\n" +
      "  \"responseHeader\": {\n" +
      "    \"QTime\": 6,\n" +
      "    \"status\": 0\n" +
      "  },\n" +
      "  \"results\": {\n" +
      "    \"command\": \"download\",\n" +
      "    \"procId\": 1476600502540,\n" +
      "    \"result\": {\n" +
      "      \"data\": {\n" +
      "        \"lucene\": [\n" +
      "          \"TF in name\",\n" +
      "          \"TF in features\",\n" +
      "          \"IDF in name\",\n" +
      "          \"IDF in features\"\n" +
      "        ],\n" +
      "        \"queries\": [\n" +
      "          {\n" +
      "            \"docs\": [\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  3.496508\n" +
      "                ],\n" +
      "                \"id\": \"docA\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  2,\n" +
      "                  1,\n" +
      "                  2.397895,\n" +
      "                  3.496508\n" +
      "                ],\n" +
      "                \"id\": \"docB\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  3.496508\n" +
      "                ],\n" +
      "                \"id\": \"docC\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  3.496508\n" +
      "                ],\n" +
      "                \"id\": \"docD\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  3.496508\n" +
      "                ],\n" +
      "                \"id\": \"docE\"\n" +
      "              }\n" +
      "            ],\n" +
      "            \"qid\": 0,\n" +
      "            \"query\": \"iPhone\"\n" +
      "          },\n" +
      "          {\n" +
      "            \"docs\": [\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  2.80336\n" +
      "                ],\n" +
      "                \"id\": \"docA\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  2.80336\n" +
      "                ],\n" +
      "                \"id\": \"docB\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  1,\n" +
      "                  0,\n" +
      "                  2.397895,\n" +
      "                  2.80336\n" +
      "                ],\n" +
      "                \"id\": \"docC\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  0,\n" +
      "                  3,\n" +
      "                  2.397895,\n" +
      "                  2.80336\n" +
      "                ],\n" +
      "                \"id\": \"docD\"\n" +
      "              },\n" +
      "              {\n" +
      "                \"lucene\": [\n" +
      "                  0,\n" +
      "                  1,\n" +
      "                  2.397895,\n" +
      "                  2.80336\n" +
      "                ],\n" +
      "                \"id\": \"docE\"\n" +
      "              }\n" +
      "            ],\n" +
      "            \"qid\": 1,\n" +
      "            \"query\": \"Android\"\n" +
      "          }\n" +
      "        ]\n" +
      "      }\n" +
      "    }\n" +
      "  }\n" +
      "}";

  @Test
  public void testGetQueryMap() throws Exception{
    Reader reader = new StringReader(testResponse);
    LTRResponseHandler parser = new LTRResponseHandler(reader);
    Map<String, LTRResponse.Doc[]> qMap = parser.getQueryMap();
    LTRResponse.Doc[] docs = qMap.get("ipod");
    Assert.assertEquals(docs.length, 3);

    LTRResponse.Doc doc = docs[0];
    Assert.assertEquals(doc.id, "F8V7067-APL-KIT");
    Assert.assertEquals(doc.lucene[0], 1, 0.01);
    Assert.assertEquals(doc.lucene[1], 0, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 3.496508, 0.01);

    doc = docs[1];
    Assert.assertEquals(doc.id, "IW-02");
    Assert.assertEquals(doc.lucene[0], 2, 0.01);
    Assert.assertEquals(doc.lucene[1], 1, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 3.496508, 0.01);

    doc = docs[2];
    Assert.assertEquals(doc.lucene[0], 1, 0.01);
    Assert.assertEquals(doc.lucene[1], 0, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 3.496508, 0.01);

    docs = qMap.get("memory");
    Assert.assertEquals(docs.length, 5);

    doc = docs[0];
    Assert.assertEquals(doc.id, "TWINX2048-3200PRO");
    Assert.assertEquals(doc.lucene[0], 1, 0.01);
    Assert.assertEquals(doc.lucene[1], 0, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 2.80336, 0.01);

    doc = docs[1];
    Assert.assertEquals(doc.id, "VS1GB400C3");
    Assert.assertEquals(doc.lucene[0], 1, 0.01);
    Assert.assertEquals(doc.lucene[1], 0, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 2.80336, 0.01);

    doc = docs[2];
    Assert.assertEquals(doc.id, "VDBDB1A16");
    Assert.assertEquals(doc.lucene[0], 1, 0.01);
    Assert.assertEquals(doc.lucene[1], 0, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 2.80336, 0.01);

    doc = docs[3];
    Assert.assertEquals(doc.id, "0579B002");
    Assert.assertEquals(doc.lucene[0], 0, 0.01);
    Assert.assertEquals(doc.lucene[1], 3, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 2.80336, 0.01);

    doc = docs[4];
    Assert.assertEquals(doc.id, "EN7800GTX/2DHTV/256M");
    Assert.assertEquals(doc.lucene[0], 0, 0.01);
    Assert.assertEquals(doc.lucene[1], 1, 0.01);
    Assert.assertEquals(doc.lucene[2], 2.397895, 0.01);
    Assert.assertEquals(doc.lucene[3], 2.80336, 0.01);
  }

  @Test
  public void testMergeClickRates() throws Exception{
    InputStream inputStream = new ByteArrayInputStream(SRC_JSON.getBytes(StandardCharsets.UTF_8));
    CMQueryHandler cmc = new CMQueryHandler(inputStream);
    LTRResponseHandler lrh = new LTRResponseHandler(new StringReader(RESPONSE_JSON));
    Map<String, Map<String, Float>> clickRates = cmc.getClickRates();

    Map<String, LTRResponse.Doc[]> mcr = lrh.mergeClickRates(clickRates);
    LTRResponse.Doc[] docs = mcr.get("iPhone");
    for(LTRResponse.Doc doc : docs)
      Assert.assertEquals((float)clickRates.get("iPhone").get(doc.id), doc.getClickrate(), 0.01);

    docs = mcr.get("Android");
    for(LTRResponse.Doc doc : docs)
      Assert.assertEquals((float)clickRates.get("Android").get(doc.id), doc.getClickrate(), 0.01);
  }
}