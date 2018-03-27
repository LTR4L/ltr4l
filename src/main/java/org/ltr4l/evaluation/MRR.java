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

import org.ltr4l.query.Document;

import java.util.List;

public class MRR implements RankEval {
  //TODO: Check if relevance can be defined
  public double calculate(List<Document> docRanks, int relevance) {
    return calculate(docRanks);
  }

  public double calculate(List<Document> docRanks){
    for (int i = 0; i < docRanks.size(); i++) if (docRanks.get(i).getLabel() > 0) return i + 1;
    return docRanks.size();
  }
}
