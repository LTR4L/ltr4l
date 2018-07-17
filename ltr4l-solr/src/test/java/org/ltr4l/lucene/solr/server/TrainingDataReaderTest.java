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

package org.ltr4l.lucene.solr.server;

import org.junit.Test;
import static org.junit.Assert.*;

public class TrainingDataReaderTest {

  private final String JSON_DATA ="{\n" +
          "  \"idField\": \"url\",\n" +
          "  \"queries\": [\n" +
          "    {\n" +
          "      \"qid\": 101,\n" +
          "      \"query\": \"サラリーマン\",\n" +
          "      \"docs\": [\n" +
          "        \"url1\",\n" +
          "        \"url2\",\n" +
          "        \"url3\"\n" +
          "      ]\n" +
          "    },\n" +
          "    {\n" +
          "      \"qid\": 102,\n" +
          "      \"query\": \"携帯電話\",\n" +
          "      \"docs\": [\n" +
          "        \"url4\",\n" +
          "        \"url5\",\n" +
          "        \"url6\",\n" +
          "        \"url7\"\n" +
          "      ]\n" +
          "    },\n" +
          "    {\n" +
          "      \"qid\": 103,\n" +
          "      \"query\": \"ラーメン\",\n" +
          "      \"docs\": [\n" +
          "        \"url8\",\n" +
          "        \"url9\",\n" +
          "        \"url10\"\n" +
          "      ]\n" +
          "    }\n" +
          "  ]\n" +
          "}\n";

  @Test
  public void testLoader() throws Exception {
    TrainingDataReader tdReader = new TrainingDataReader(JSON_DATA);
    assertEquals("url", tdReader.getIdField());
    TrainingDataReader.QueryDataDesc[] queryDataDescs = tdReader.getQueryDataDescs();
    assertEquals(3, queryDataDescs.length);
    assertEquals("qid=101,query=サラリーマン,docs=[url1,url2,url3]", queryDataDescs[0].toString());
    assertEquals("qid=102,query=携帯電話,docs=[url4,url5,url6,url7]", queryDataDescs[1].toString());
    assertEquals("qid=103,query=ラーメン,docs=[url8,url9,url10]", queryDataDescs[2].toString());
  }

  @Test
  public void testGetTotalDocs() throws Exception {
    TrainingDataReader tdReader = new TrainingDataReader(JSON_DATA);
    assertEquals(10, tdReader.getTotalDocs());
  }
}
