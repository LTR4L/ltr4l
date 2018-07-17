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

import org.apache.solr.SolrTestCaseJ4;
import org.apache.solr.common.SolrException;
import org.apache.solr.common.params.ModifiableSolrParams;
import org.apache.solr.common.util.ContentStream;
import org.apache.solr.common.util.ContentStreamBase;
import org.apache.solr.common.util.SimpleOrderedMap;
import org.apache.solr.core.SolrCore;
import org.apache.solr.request.LocalSolrQueryRequest;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;
import org.apache.solr.update.processor.BufferingRequestProcessor;
import org.apache.solr.util.TestHarness;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import static org.apache.solr.util.TestHarness.*;

public class FeaturesRequestHandlerTest extends SolrTestCaseJ4 {

  @BeforeClass
  public static void beforeClass() throws Exception {
    initCore("conf/solrconfig.xml", "conf/schema.xml", "src/test/resources");
  }

  @Override
  @Before
  public void setUp() throws Exception {
    // if you override setUp or tearDown, you better call
    // the super classes version
    super.setUp();
    clearIndex();
    assertU(commit());
  }

  @Test
  public void test() throws Exception {
    assertU(adoc("id", "1", "title", "this is title", "body", "this is body"));
    assertU(commit());

    /*
     * command=extract
     */
    SolrQueryRequest req = req("command", "extract", "conf", "ltr_features.conf");
    FeaturesRequestHandler handler = new FeaturesRequestHandler();
    SimpleOrderedMap<Object> results = new SimpleOrderedMap<Object>();
    handler.handleExtract(req,
            makeStream("{\n" +
                    "  \"idField\": \"id\",\n" +
                    "  \"queries\": [\n" +
                    "    {\n" +
                    "      \"qid\": 101,\n" +
                    "      \"query\": \"this\",\n" +
                    "      \"docs\": [ \"1\" ]\n" +
                    "    },\n" +
                    "    {\n" +
                    "      \"qid\": 102,\n" +
                    "      \"query\": \"title\",\n" +
                    "      \"docs\": [ \"1\" ]\n" +
                    "    },\n" +
                    "    {\n" +
                    "      \"qid\": 103,\n" +
                    "      \"query\": \"body\",\n" +
                    "      \"docs\": [ \"1\" ]\n" +
                    "    }\n" +
                    "  ]\n" +
                    "}\n"), results);

    long procId = (Long)results.get("procId");

    req.close();

    /*
     * command=progress
     */
    SolrQueryRequest req2 = req("command", "progress", "procId", Long.toString(procId));
    results = new SimpleOrderedMap<Object>();
    handler.handleProgress(req2, results);

    long procId2 = (Long)results.get("procId");
    assertEquals(procId, procId2);
    assertNotNull(results.get("done?"));
    assertNotNull(results.get("progress"));

    req2.close();

    /*
     * command=delete
     */
    SolrQueryRequest req3 = req("command", "delete", "procId", Long.toString(procId));
    results = new SimpleOrderedMap<Object>();
    handler.handleDelete(req3, results);

    long procId3 = (Long)results.get("procId");
    assertEquals(procId, procId3);
    assertNotNull(results.get("result"));

    req3.close();

    /*
     * command=progress => procId is no longer valid
     */
    SolrQueryRequest req4 = req("command", "progress", "procId", Long.toString(procId));
    results = new SimpleOrderedMap<Object>();
    try {
      handler.handleProgress(req4, results);
      fail("this method should fail because the process is no longer valid");
    }
    catch (SolrException expected){}
    finally {
      req4.close();
    }
  }

  private Iterable<ContentStream> makeStream(String json){
    List<ContentStream> result = new ArrayList<ContentStream>();
    result.add(new ContentStreamBase.StringStream(json, "application/json"));
    return result;
  }
}
