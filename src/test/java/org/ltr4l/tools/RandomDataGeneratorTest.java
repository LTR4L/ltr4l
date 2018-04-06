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

package org.ltr4l.tools;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

public class RandomDataGeneratorTest {

  @Test
  public void testAlpha() throws Exception {
    Assert.assertEquals(0.2, RandomDataGenerator.alpha(1, 2), 0.0001);
    Assert.assertEquals(0.1, RandomDataGenerator.alpha(1, 3), 0.0001);
    Assert.assertEquals(0.05, RandomDataGenerator.alpha(1, 4), 0.0001);

    Assert.assertEquals(0.15, RandomDataGenerator.alpha(2, 2), 0.0001);
    Assert.assertEquals(0.075, RandomDataGenerator.alpha(2, 3), 0.0001);
    Assert.assertEquals(0.0375, RandomDataGenerator.alpha(2, 4), 0.0001);

    Assert.assertEquals(0.1, RandomDataGenerator.alpha(3, 2), 0.0001);
    Assert.assertEquals(0.05, RandomDataGenerator.alpha(3, 3), 0.0001);
    Assert.assertEquals(0.025, RandomDataGenerator.alpha(3, 4), 0.0001);

    Assert.assertEquals(0.05, RandomDataGenerator.alpha(4, 2), 0.0001);
    Assert.assertEquals(0.025, RandomDataGenerator.alpha(4, 3), 0.0001);
    Assert.assertEquals(0.0125, RandomDataGenerator.alpha(4, 4), 0.0001);
  }

  @Test
  public void testGenerateD1S2() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(1, 2);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 10, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
  }

  @Test
  public void testGenerateD1S3() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(1, 3);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 20, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
  }

  @Test
  public void testGenerateD2S2() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(2, 2);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 10, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
  }

  @Test
  public void testGenerateD2S3() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(2, 3);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 20, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
  }

  @Test
  public void testGenerateD3S2() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(3, 2);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 10, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 10, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(10, documents.size());
    assertContents(documents, 2, 2);
  }

  @Test
  public void testGenerateD3S3() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(3, 3);

    // num of query = 1
    QuerySet querySet = rdg.getRandomQuerySet(1, 20, 2);
    List<Query> queries = querySet.getQueries();
    Assert.assertEquals(1, queries.size());

    List<Document> documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 2
    querySet = rdg.getRandomQuerySet(2, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(2, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);

    // num of query = 3
    querySet = rdg.getRandomQuerySet(3, 20, 2);
    queries = querySet.getQueries();
    Assert.assertEquals(3, queries.size());

    documents = queries.get(0).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(1).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
    documents = queries.get(2).getDocList();
    Assert.assertEquals(20, documents.size());
    assertContents(documents, 3, 2);
  }

  private static void assertContents(List<Document> documents, int stars, int minSamples){
    Map<Integer, Integer> map = new HashMap<>();
    for(Document document: documents){
      Integer count = map.get(document.getLabel());
      if(count == null){
        map.put(document.getLabel(), 1);
      }
      else{
        map.put(document.getLabel(), count + 1);
      }
    }

    Assert.assertEquals(stars, map.size());

    for(int star: map.keySet()){
      Assert.assertTrue(map.get(star) >= minSamples);
    }
  }
}
