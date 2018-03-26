/*
 * Copyright 2018 org.LTR4L
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
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


}