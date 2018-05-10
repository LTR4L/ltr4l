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
package org.ltr4l.query;

import org.junit.Assert;
import org.junit.Test;

import java.io.*;
import java.util.List;

public class QuerySetTest {

  @Test (expected = NumberFormatException.class)
  public void testParseQueriesInvalid() throws Exception{
    StringReader sr = new StringReader("0 qid:1 1:5.000000ab 2:3.465736 3:0.500000 4:0.476551 #docid = 244338\n" +
        "10 qid:10035 1:0.356436 2:0.750000 3:0.428571 4:0.000000 #docid = GX072-27-16566993 inc = 0.00370129850418619 prob = 0.0897079"
    );
    QuerySet querySet = new QuerySet();
    querySet.parseQueries(sr);
  }

  @Test (expected = ArrayIndexOutOfBoundsException.class)
  public void testParseQueriesFormat() throws Exception{
    StringReader sr = new StringReader("0 qid1 15.000000ab 23.465736 30.500000 4:0.476551 #docid = 244338"); //No regex ":"
    QuerySet querySet = new QuerySet();
    querySet.parseQueries(sr);
  }

  @Test
  public void testParseQueries() throws Exception{
    //String testFile = "fooDataTest.txt";
    double[] features1 = {5.000000, 3.465736, 0.500000, 0.476551};
    double[] features2 = {0.356436, 0.750000, 0.428571, 0.000000};
    StringReader sr = new StringReader("0 qid:1 1:5.000000 2:3.465736 3:0.500000 4:0.476551 #docid = 244338\n" +
        "10 qid:10035 1:0.356436 2:0.750000 3:0.428571 4:0.000000 #docid = GX072-27-16566993 inc = 0.00370129850418619 prob = 0.0897079"
    );
    QuerySet querySet = new QuerySet();
    querySet.parseQueries(sr);
    sr.close();
    List<Query> queries = querySet.getQueries();
    Query query1 = queries.get(0);
    Query query2 = queries.get(1);
    List<Document> docs1 = query1.getDocList();
    List<Document> docs2 = query2.getDocList();
    Assert.assertEquals(2, queries.size());
    Assert.assertEquals(1, query1.getQueryId());
    Assert.assertEquals(10035, query2.getQueryId());
    Assert.assertEquals(features1.length, query1.getFeatureLength());
    Assert.assertEquals(features2.length, query2.getFeatureLength());
    Assert.assertEquals(1, docs1.size());
    Assert.assertEquals(1, docs2.size());
    Assert.assertEquals(0, docs1.get(0).getLabel());
    Assert.assertEquals(10, docs2.get(0).getLabel());
    for (int i = 0; i < 4; i++) {
      Assert.assertEquals(features1[i], docs1.get(0).getFeature(i), 0.0001);
      Assert.assertEquals(features2[i], docs2.get(0).getFeature(i), 0.0001);
    }
  }

  @Test
  public void testFindMaxLabel() throws Exception {
    StringReader sr = new StringReader("0 qid:1 1:5.000000 2:3.465736 3:0.500000 4:0.476551 #docid = 244338\n" +
        "10 qid:10035 1:0.356436 2:0.750000 3:0.428571 4:0.000000 #docid = GX072-27-16566993 inc = 0.00370129850418619 prob = 0.0897079"
    );
    QuerySet querySet = new QuerySet();
    querySet.parseQueries(sr);
    sr.close();
    Assert.assertEquals(10, QuerySet.findMaxLabel(querySet.getQueries()));
  }
}