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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;
import org.ltr4l.tools.Config;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class RankBoost extends Ranker<RankBoost.RankBoostConfig> {
  private final List<WeakLearner> learners;

  public RankBoost(){
    learners = new ArrayList<>();
  }

  public RankBoost(Reader reader){
    learners = readModel(reader);
  }

  public void addLearner(WeakLearner wl){
    learners.add(wl);
  }

  @Override
  public void writeModel(RankBoostConfig config, Writer writer) throws IOException {
    int[] features = learners.stream().mapToInt(l -> l.getFid()).toArray();
    double[] thresholds = learners.stream().mapToDouble(l -> l.getThreshold()).toArray();
    double[] alphas = learners.stream().mapToDouble(l -> l.getAlpha()).toArray();
    SavedModel model = new SavedModel(config, features, thresholds, alphas);
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, model);
  }

  public List<WeakLearner> readModel(Reader reader){
    List<WeakLearner> wls = new ArrayList<>();
    try{
      Objects.requireNonNull(reader);
      ObjectMapper mapper = new ObjectMapper();
      SavedModel model = mapper.readValue(reader, SavedModel.class);
      model.assertLengths();
      for(int i = 0; i < model.thresholds.length; i++)
        wls.add(new WeakLearner(model.features[i], model.thresholds[i], model.weights[i]));
    } catch(IOException e){
      throw new RuntimeException();
    }
    return wls;
  }

  @Override
  public double predict(List<Double> features) {
    return learners.isEmpty() ? 0 : learners.stream().mapToDouble(wl -> wl.getAlpha() * wl.predict(features)).sum();
  }

  public static class RankBoostConfig extends Config {
    @JsonIgnore
    public int getNumSteps() { return getInt(params, "numSteps", 0); } //TODO: OK default value?
  }

  protected static class SavedModel {
    public RankBoostConfig config;
    public int[] features;
    public double[] thresholds;
    public double[] weights; //alphas

    public SavedModel(){ //this is needed for Jackson...
    }

    public SavedModel(RankBoostConfig config, int[] features, double[] thresholds, double[] weights){
      assert(features.length == thresholds.length && thresholds.length == weights.length);
      this.config = config;
      this.features = features;
      this.thresholds = thresholds;
      this.weights = weights;
    }

    @JsonIgnore
    public void assertLengths(){
      assert(features.length == thresholds.length && thresholds.length == weights.length);
    }

  }
}
