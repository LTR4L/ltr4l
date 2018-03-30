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

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;

/**
 * The implementation of LTRTrainer which uses the
 * PRank(Perceptron Ranking) algorithm.
 *
 */
public class PRankTrainer extends LTRTrainer<PRank, Config> {

  private final  List<Document> trainingDocList;

  public static PRank getPRank(int featureLength, int maxLabel){
    return new PRank(featureLength, maxLabel);
  }

  PRankTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override);
    maxScore = 0.0;
    trainingDocList = new ArrayList<>();
    for (Query query : trainingSet)
      trainingDocList.addAll(query.getDocList());
  }

  @Override
  public void train() {
    Collections.shuffle(trainingDocList);
    for (Document doc : trainingDocList)
      ranker.updateWeights(doc);
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Square();
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
  public Class<Config> getConfigClass() {
    return Config.class;
  }

  @Override
  protected PRank constructRanker() {
    return new PRank(trainingSet.get(0).getFeatureLength(), QuerySet.findMaxLabel(trainingSet));
  }
}

class PRank extends Ranker<Config> {
  protected double[] weights;
  protected double[] thresholds;

  PRank(int featureLength, int maxLabel) {
    if (featureLength > 0 && maxLabel > 0) {
      weights = new double[featureLength];
      thresholds = new double[maxLabel];
    } else {
      weights = null;
      thresholds = null;
    }

  }

  public double[] getWeights() {
    return weights;
  }

  public double[] getThresholds() {
    return thresholds;
  }

  @Override
  public void writeModel(Config config, Writer writer) throws IOException {
    SavedModel savedModel = new SavedModel(config, weights, thresholds);
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, savedModel);
  }

  // TODO: use Factory...?
  public static PRank readModel(Reader reader) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    SavedModel savedModel = mapper.readValue(reader, SavedModel.class);
    // TODO: don't want to do that...
    PRank prank = new PRank(savedModel.weights.length, savedModel.thresholds.length);
    prank.weights = savedModel.weights;
    prank.thresholds = savedModel.thresholds;
    return prank;
  }

  public void updateWeights(Document doc) {
    double wx = predictRelScore(doc.getFeatures());
    int output = (int) predict(doc.getFeatures());
    int label = doc.getLabel();
    if (output == label)//if output == label, do not update weights.
      return;
    int[] tau = new int[thresholds.length];
    for (int r = 0; r <= thresholds.length - 1; r++) { /////thresholds.length ??
      int ytr;
      if (label <= r)
        ytr = -1;
      else
        ytr = 1;
      if ((wx - thresholds[r]) * ytr <= 0)
        tau[r] = ytr;
      else
        tau[r] = 0;
    }
    int T = IntStream.of(tau).sum();
    for (int i = 0; i <= weights.length - 1; i++) {
      weights[i] += T * doc.getFeatures().get(i);
    }
    for (int r = 0; r <= thresholds.length - 1; r++) {
      thresholds[r] -= tau[r];
    }
  }


  @Override
  public double predict(List<Double> features) {
    double wx = predictRelScore(features);
    for (int i = 0; i < thresholds.length; i++) {
      double b = thresholds[i];
      if (wx < b)
        return i;
    }
    return thresholds.length;
  }

  private double predictRelScore(List<Double> features){
    double wx = 0;
    for (int i = 0; i < features.size(); i++){
      wx += features.get(i) * weights[i];
    }
    return wx;
  }

  private static class SavedModel {
    public Config config;
    public double[] weights;
    public double[] thresholds;
    SavedModel(){  // this is needed for Jackson...
    }
    SavedModel(Config config, double[] weights, double[] thresholds){
      this.config = config;
      this.weights = weights;
      this.thresholds = thresholds;
    }
  }
}