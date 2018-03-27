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

package org.ltr4l.trainers;

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import org.ltr4l.Ranker;
import org.ltr4l.nn.OAPBPMRank;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;

/**
 * The implementation of LTRTrainer which uses the
 * OAP-BPM algorithm.
 *
 */
public class OAPBPMTrainer extends LTRTrainer<OAPBPMRank, OAPBPMTrainer.OAPBPMConfig> {
  private double maxScore;
  private final  List<Document> trainingDocList;

  OAPBPMTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override);
    maxScore = 0d;
    trainingDocList = new ArrayList<>();
    for (Query query : trainingSet)
      trainingDocList.addAll(query.getDocList());
  }

  @Override
  public void train() {
    for (Document doc : trainingDocList)
      ranker.updateWeights(doc);
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Square();
  }

  @Override
  public Class<OAPBPMConfig> getConfigClass() {
    return getCC();
  }

  static Class<OAPBPMConfig> getCC(){
    return OAPBPMConfig.class;
  }

  protected double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> errorFunc.error(ranker.predict(doc.getFeatures()), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  @Override
  protected Ranker constructRanker() {
    return new OAPBPMRank(trainingSet.get(0).getFeatureLength(), QuerySet.findMaxLabel(trainingSet), config.getPNum(), config.getBernNum());
  }

  public static class OAPBPMConfig extends Config {

    public int getPNum(){
      return getInt(params, "N", 1);   // TODO: default value 1 is appropriate?
    }

    public double getBernNum(){
      return getDouble(params, "bernoulli", 0.03);
    }
  }
}

