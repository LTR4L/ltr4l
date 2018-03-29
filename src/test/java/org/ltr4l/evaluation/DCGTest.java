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

package org.ltr4l.evaluation;

import java.util.ArrayList;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;

public class DCGTest {

  @Test
  public void testDCGPerfectMatch() throws Exception {
    List<Document> documents = docs(3, 3, 2, 2, 1, 1, 1);
    Assert.assertEquals(7, DCG.dcg(documents, 1), 0.001);
    Assert.assertEquals(11.41650828, DCG.dcg(documents, 2), 0.001);
    Assert.assertEquals(12.91650828, DCG.dcg(documents, 3), 0.001);
    Assert.assertEquals(14.20853795, DCG.dcg(documents, 4), 0.001);
    Assert.assertEquals(14.59539076, DCG.dcg(documents, 5), 0.001);
    Assert.assertEquals(14.95159794, DCG.dcg(documents, 6), 0.001);
    Assert.assertEquals(15.28493128, DCG.dcg(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(15.28493128, DCG.dcg(documents, 8), 0.001);
    Assert.assertEquals(15.28493128, DCG.dcg(documents, 9), 0.001);
    Assert.assertEquals(15.28493128, DCG.dcg(documents, 10), 0.001);
  }

  @Test
  public void testDCGImerfectMatch() throws Exception {
    List<Document> documents = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(7, DCG.dcg(documents, 1), 0.001);
    Assert.assertEquals(8.892789261, DCG.dcg(documents, 2), 0.001);
    Assert.assertEquals(12.39278926, DCG.dcg(documents, 3), 0.001);
    Assert.assertEquals(13.68481893, DCG.dcg(documents, 4), 0.001);
    Assert.assertEquals(14.07167174, DCG.dcg(documents, 5), 0.001);
    Assert.assertEquals(14.42787893, DCG.dcg(documents, 6), 0.001);
    Assert.assertEquals(14.76121226, DCG.dcg(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(14.76121226, DCG.dcg(documents, 8), 0.001);
    Assert.assertEquals(14.76121226, DCG.dcg(documents, 9), 0.001);
    Assert.assertEquals(14.76121226, DCG.dcg(documents, 10), 0.001);
  }

  @Test
  public void testNDCGPerfectMatch() throws Exception {
    DCG.NDCG evaluator = new DCG.NDCG();
    List<Document> documents = docs(3, 3, 2, 2, 1, 1, 1);
    Assert.assertEquals(1, evaluator.calculate(documents, 1), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 2), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 3), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 4), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 5), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 6), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(1, evaluator.calculate(documents, 8), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 9), 0.001);
    Assert.assertEquals(1, evaluator.calculate(documents, 10), 0.001);
  }

  @Test
  public void testNDCGImperfectMatch() throws Exception {
    DCG.NDCG evaluator = new DCG.NDCG();
    List<Document> documents1 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(1, evaluator.calculate(documents1, 1), 0.001);

    // as document list is sorted in calculate, it must be re-instantiated again
    List<Document> documents2 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.778941253, evaluator.calculate(documents2, 2), 0.001);

    List<Document> documents3 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.959453515, evaluator.calculate(documents3, 3), 0.001);

    List<Document> documents4 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.963140542, evaluator.calculate(documents4, 4), 0.001);

    List<Document> documents5 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.964117506, evaluator.calculate(documents5, 5), 0.001);

    List<Document> documents6 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.964972372, evaluator.calculate(documents6, 6), 0.001);

    List<Document> documents7 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.965736253, evaluator.calculate(documents7, 7), 0.001);

    // position is greater than documents.size()
    List<Document> documents8 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.965736253, evaluator.calculate(documents8, 8), 0.001);

    List<Document> documents9 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.965736253, evaluator.calculate(documents9, 9), 0.001);

    List<Document> documents10 = docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(0.965736253, evaluator.calculate(documents10, 10), 0.001);
  }

  static List<Document> docs(int... labels){
    List<Document> documents = new ArrayList<>();
    for(int label: labels){
      Document doc = new Document();
      doc.setLabel(label);
      documents.add(doc);
    }
    return documents;
  }
}
