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

import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.trainers.Trainer;

public interface RankEval {
  static int countNumRelDocs(List<Document> docRanks){
    return docRanks.stream().filter(doc -> doc.getLabel() != 0).mapToInt(doc -> 1).sum();
  }

  static double cg(List<Document> docRanks, int position){
    assert(position > 0);
    double cg = 0;
    int pos = Math.min(position, docRanks.size());
    for (int k = 0; k < pos; k++) cg += docRanks.get(k).getLabel(); //Modified version: Math.pow(2, docRanks.get(k).getLabel()) - 1;
    return cg;
  }

  default double calculateAvgAllQueries(Ranker ranker, List<Query> queries, int position){
    double total = 0;
    for (Query query : queries) {
      double queryVal = calculate(ranker.sort(query), position);
      if (!Double.isFinite(queryVal)) continue;
      total += queryVal;
    }
    return total / queries.size();
  }

  default int identity(Document doc){
    return doc.getLabel() > 0 ? 1 : 0;
  }

  double calculate(List<Document> docRanks, int position);

  public static class RankEvalFactory {
    public static RankEval get(String eval){
      switch (eval.toLowerCase()){
        case "dcg":
          return new DCG();
        case "ndcg":
          return new DCG.NDCG();
        case "precision":
          return new Precision();
        case "map":
          return new Precision.AP();
        case "wap":
          return new Precision.WAP();
        case "mrr":
          return new MRR();
        default:
          throw new IllegalArgumentException("Invalid evaluation type specified.");
      }
    }
  }
}
