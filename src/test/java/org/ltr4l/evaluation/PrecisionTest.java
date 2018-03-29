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

import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;

public class PrecisionTest {

  @Test
  public void testPrecisionAllRelevant() throws Exception {
    List<Document> documents = DCGTest.docs(3, 2, 3, 2, 1, 1, 1);
    Assert.assertEquals(1, Precision.precision(documents, 1), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 2), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 3), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 4), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 5), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 6), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(1, Precision.precision(documents, 8), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 9), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 10), 0.001);
  }

  @Test
  public void testPrecisionIrrelevantPerfectMatch() throws Exception {
    List<Document> documents = DCGTest.docs(3, 3, 2, 2, 1, 0, 0);
    Assert.assertEquals(1, Precision.precision(documents, 1), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 2), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 3), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 4), 0.001);
    Assert.assertEquals(1, Precision.precision(documents, 5), 0.001);
    Assert.assertEquals(0.833333333, Precision.precision(documents, 6), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(0.714285714, Precision.precision(documents, 8), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 9), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 10), 0.001);
  }

  @Test
  public void testPrecisionIrrelevantImperfectMatch() throws Exception {
    List<Document> documents = DCGTest.docs(0, 3, 2, 3, 2, 0, 1);
    Assert.assertEquals(0, Precision.precision(documents, 1), 0.001);
    Assert.assertEquals(0.5, Precision.precision(documents, 2), 0.001);
    Assert.assertEquals(0.666666666, Precision.precision(documents, 3), 0.001);
    Assert.assertEquals(0.75, Precision.precision(documents, 4), 0.001);
    Assert.assertEquals(0.80, Precision.precision(documents, 5), 0.001);
    Assert.assertEquals(0.666666666, Precision.precision(documents, 6), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(0.714285714, Precision.precision(documents, 8), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 9), 0.001);
    Assert.assertEquals(0.714285714, Precision.precision(documents, 10), 0.001);
  }

  @Test
  public void testPrecisionToBinaryRelevance(){
    List<Document> documents = DCGTest.docs(0, 3, 2, 3, 2, 0, 1);
    List<Document> documents2 = DCGTest.docs(0, 1, 1, 1, 1, 0, 1);
    Assert.assertEquals(Precision.precision(documents, 1), Precision.precision(documents2, 1), 0.001);
    Assert.assertEquals(Precision.precision(documents, 2), Precision.precision(documents2, 2), 0.001);
    Assert.assertEquals(Precision.precision(documents, 3), Precision.precision(documents2, 3), 0.001);
    Assert.assertEquals(Precision.precision(documents, 4), Precision.precision(documents2, 4), 0.001);
    Assert.assertEquals(Precision.precision(documents, 5), Precision.precision(documents2, 5), 0.001);
    Assert.assertEquals(Precision.precision(documents, 6), Precision.precision(documents2, 6), 0.001);
    Assert.assertEquals(Precision.precision(documents, 7), Precision.precision(documents2, 7), 0.001);

    // position is greater than documents.size()
    Assert.assertEquals(Precision.precision(documents, 8), Precision.precision(documents2, 8), 0.001);
    Assert.assertEquals(Precision.precision(documents, 9), Precision.precision(documents2, 9), 0.001);
    Assert.assertEquals(Precision.precision(documents, 10), Precision.precision(documents2, 10), 0.001);
  }

  @Test
  public void testAPPerfectMatch() throws Exception {
    //Perfect match means all relevant documents are ranked higher up.
    List<Document> documents = DCGTest.docs(3, 2, 3, 2, 1, 0, 0);
    Precision.AP ap = new Precision.AP();
    Assert.assertEquals(1, ap.calculate(documents), 0.001);
  }

  @Test
  public void testMAPImperfectMatch() throws Exception {
    List<Document> documents = DCGTest.docs(0, 3, 2, 3, 2, 0, 1);
    Precision.AP map = new Precision.AP();
    Assert.assertEquals(0.686190476, map.calculate(documents), 0.001);
    //Show equivalence with binary case
    Assert.assertEquals(map.calculate(documents), map.calculate(DCGTest.docs(0, 1, 1, 1, 1, 0, 1)), 0.001);
  }

  @Test
  public void testWAPPerfectMatch() throws Exception {
    //Perfect match means all relevant documents are ranked higher up.
    List<Document> documents = DCGTest.docs(3, 3, 2, 2, 1, 0, 0);
    Precision.WAP wap = new Precision.WAP();
    Assert.assertEquals(1, wap.calculate(documents), 0.001);
  }

  @Test
  public void testWAPImperfectMatch() throws Exception {
    List<Document> documents = DCGTest.docs(0, 3, 2, 3, 2, 0, 1);
    Precision.WAP wap = new Precision.WAP();
    Assert.assertEquals(0.766818, wap.calculate(documents), 0.001);
  }

}