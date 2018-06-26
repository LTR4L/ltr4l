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

public class RankBoost extends AdaBoost {

  public RankBoost(){
    super();
  }

  public RankBoost(Reader reader){
    super(reader);
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
