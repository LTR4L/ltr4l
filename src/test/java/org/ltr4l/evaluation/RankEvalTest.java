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

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.evaluation.RankEval.*;

import java.util.List;

public class RankEvalTest {

  @Test
  public void testCountNumRelDocs() {
    List<Document> docs = DCGTest.docs(3, 3, 2, 1, 1, 0, 0);
    Assert.assertEquals(5, RankEval.countNumRelDocs(docs));
  }

  @Test
  public void testCountNoRelevant() {
    List<Document> docs = DCGTest.docs(0, 0, 0, 0, 0, 0, 0, 0);
    Assert.assertEquals(0, RankEval.countNumRelDocs(docs));
  }

  @Test
  public void testCG(){
    List<Document> docs = DCGTest.docs(3, 3, 2, 1, 1, 0, 0);
    Assert.assertEquals(3, RankEval.cg(docs, 1), 0.001);
    Assert.assertEquals(6, RankEval.cg(docs, 2), 0.001);
    Assert.assertEquals(8, RankEval.cg(docs, 3), 0.001);
    Assert.assertEquals(9, RankEval.cg(docs, 4), 0.001);
    Assert.assertEquals(10, RankEval.cg(docs, 5), 0.001);
    Assert.assertEquals(10, RankEval.cg(docs, 6), 0.001);

    //position greater than docs.size()
    Assert.assertEquals(10, RankEval.cg(docs, 7), 0.001);
    Assert.assertEquals(10, RankEval.cg(docs, 8), 0.001);
    Assert.assertEquals(10, RankEval.cg(docs, 9), 0.001);
  }

  @Test (expected = AssertionError.class)
  public void testCGInvalidPosition(){
    List<Document> docs = DCGTest.docs(0, 0, 0, 0, 0, 0);
    RankEval.cg(docs, -1);
  }

  @Test
  public void factoryTest() {
    Assert.assertTrue(RankEvalFactory.get("NDCG") instanceof DCG.NDCG);
    Assert.assertTrue(RankEvalFactory.get("MAP") instanceof Precision.AP);
    Assert.assertTrue(RankEvalFactory.get("MRR") instanceof MRR);
    Assert.assertTrue(RankEvalFactory.get("WAP") instanceof Precision.WAP);
    Assert.assertTrue(RankEvalFactory.get("DCG") instanceof DCG);
    Assert.assertTrue(RankEvalFactory.get("Precision") instanceof Precision);
  }

  @Test(expected = IllegalArgumentException.class)
  public void factoryBadOptionTest(){
    RankEvalFactory.get("MyGreatestEvaluator!");
  }

}