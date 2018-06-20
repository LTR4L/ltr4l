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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class AdaBoost extends Ranker<RankBoost.RankBoostConfig> {
  protected final List<WeakLearner> learners;

  public AdaBoost(){
    learners = new ArrayList<>();
  }

  @Override
  public void writeModel(RankBoost.RankBoostConfig config, Writer writer) throws IOException {
    int[] features = learners.stream().mapToInt(l -> l.getFid()).toArray();
    double[] thresholds = learners.stream().mapToDouble(l -> l.getThreshold()).toArray();
    double[] alphas = learners.stream().mapToDouble(l -> l.getAlpha()).toArray();
    RankBoost.SavedModel model = new RankBoost.SavedModel(config, features, thresholds, alphas);
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, model);
  }

  @Override
  public double predict(List<Double> features) { //Adaboost predictions are binary
     return Math.signum(learners.stream().mapToDouble(learner -> learner.predict(features)).sum());
  }

  public List<WeakLearner> readModel(Reader reader){
    List<WeakLearner> wls = new ArrayList<>();
    try{
      Objects.requireNonNull(reader);
      ObjectMapper mapper = new ObjectMapper();
      RankBoost.SavedModel model = mapper.readValue(reader, RankBoost.SavedModel.class);
      model.assertLengths();
      for(int i = 0; i < model.thresholds.length; i++)
        wls.add(new WeakLearner(model.features[i], model.thresholds[i], model.weights[i]));
    } catch(IOException e){
      throw new RuntimeException(e);
    }
    return wls;
  }
}
