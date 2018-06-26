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
package org.ltr4l.boosting;

import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.List;

public class ABDistribution extends Distribution.D2 {
  protected double normFactor;

  public ABDistribution(List<RankedDocs> queries){ //TODO: doesn't have to be RankedDocs for AdaBoost...
    super(queries);
  }

  @Override
  protected void initialize(List<RankedDocs> queries){
    int totalDocs = queries.stream().mapToInt(q -> q.getRankedDocs().stream().mapToInt(doc -> 1).sum()).sum();
    for(int i = 0; i < queries.size(); i++){
      List<Document> queryDocs = queries.get(i).getRankedDocs();
      dist[i] = new double[queryDocs.size()];
      for(int j = 0; j < queryDocs.size(); j++)
        dist[i][j] = 1d/totalDocs;
    }
  }

  public void update(WeakLearner wl, List<RankedDocs> queries ){ //TODO: Does not have to be ordered; matched with RBDist
    double newNormFactor = 0d;
    for(int qid = 0; qid < queries.size(); qid++){
      newNormFactor += updateQuery(wl, qid, queries.get(qid).getRankedDocs());
    }
    normalize(newNormFactor);
    normFactor = newNormFactor;
  }

  @Override
  protected double updateQuery(WeakLearner wl, int qid, List<Document> queryDocs){ //returns the query normalization factor
    double newNormFactor = 0d;
    for(int i = 0; i < queryDocs.size() - 1; i++){
      Document doc = queryDocs.get(i);
      //Adaboost distribution considers only two classes; i.e. relevant or irrelevant
      dist[qid][i] *= Math.exp(-wl.getAlpha() * (doc.getLabel() <= 0 ? -1 : 1) * wl.predict(doc.getFeatures()));
      newNormFactor += dist[qid][i];
    }
    return newNormFactor;
  }

}
