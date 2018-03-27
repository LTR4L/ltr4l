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

public class MRRTest {
  @Test
  public void hasRelevant(){
    List<Document> documents = DCGTest.docs(0, 0, 3, 2, 1);
    MRR mrr = new MRR();
    Assert.assertEquals(3, mrr.calculate(documents), 0.01);
  }

  @Test
  public void hasNoRelevant(){
    List<Document> documents = DCGTest.docs(0, 0, 0, 0, 0, 0);
    MRR mrr = new MRR();
    Assert.assertEquals(6, mrr.calculate(documents), 0.01);
  }

}