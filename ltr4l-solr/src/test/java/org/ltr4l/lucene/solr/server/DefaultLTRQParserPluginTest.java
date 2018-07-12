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
import org.apache.solr.common.params.ModifiableSolrParams;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;

public class DefaultLTRQParserPluginTest extends SolrTestCaseJ4 {

  @BeforeClass
  public static void beforeClass() throws Exception {
    initCore("solrconfig.xml", "schema.xml", "src/test/resources");
  }

  @Override
  @Before
  public void setUp() throws Exception {
    super.setUp();
    clearIndex();
    assertU(commit());
  }

  @Test
  public void testDirectUse() throws Exception {
    assertU(adoc("id", "1", "title", "this is title", "body", "this is body"));
    assertU(commit());

    ModifiableSolrParams params = new ModifiableSolrParams();
    params.add("q", "{!nn}title body").add("fl", "*,score");
    assertQ(req(params, "indent", "on"), "*[count(//doc)=1]",
            "//result/doc[1]/str[@name='id'][.='1']",
            "//result/doc[1]/float[@name='score'][.='-0.015751183']"
    );
  }

  @Test
  public void testReRank() throws Exception {
    assertU(adoc("id", "1", "title", "this is title", "body", "this is body"));
    assertU(commit());

    ModifiableSolrParams params = new ModifiableSolrParams();
    params.add("q", "{!rerank reRankQuery=$rqq reRankWeight=2.0}title body");
    params.add("rqq", "{!nn}title body").add("fl","*,score");
    assertQ(req(params, "indent", "on"), "*[count(//doc)=1]",
            "//result/doc[1]/str[@name='id'][.='1']",
            "//result/doc[1]/float[@name='score'][.='0.96849763']" // 1.0 - (0.015751183 * 2)
    );
  }
}
