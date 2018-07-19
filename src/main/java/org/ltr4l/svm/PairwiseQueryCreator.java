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
package org.ltr4l.svm;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class PairwiseQueryCreator {

  private PairwiseQueryCreator(){}

  public static List<Query> createQueries(List<Query> origQueries){
    Document[][][] pairs = origQueries.stream().map(q -> q.orderDocPairs()).toArray(Document[][][]::new);
    //qid info lost in above; only index remains
    List<Query> pwQueries = new ArrayList<>();
    for (Document[][] origQuery : pairs){
      Query nQuery = createQuery(origQuery);
      pwQueries.add(nQuery);
    }
    return pwQueries;
  }

  private static Query createQuery(Document[][] origQuery){
    // Document[][] --> [pair#][0 or 1; which doc in pair]
    Query query = new Query();
    for (Document[] pair : origQuery){
      assert(pair[0].getLabel() > pair[1].getLabel());
      List<Double> feat0 = pair[0].getFeatures();
      List<Double> feat1 = pair[1].getFeatures();
      Document doc1 = new Document(VectorMath.diff(feat0, feat1), 1);
      Document doc2 = new Document(VectorMath.diff(feat1, feat0), -1);
      query.addDocument(doc1);
      query.addDocument(doc2);
    }
    Collections.shuffle(query.getDocList());
    return query;
  }

}
