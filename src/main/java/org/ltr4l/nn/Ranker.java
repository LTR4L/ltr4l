/*
 * Copyright 2018 org.LTR4L
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.ltr4l.nn;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Properties;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Ranker classes use a model to make predictions for a document.
 * Models (weights, thresholds, etc...) held by Ranker classes
 * can be trained by trainers.
 *
 * Rankers are the model holders.
 */
public abstract class Ranker {
  protected static final String DEFAULT_MODEL_FILE = "model.txt";

  public abstract void writeModel(Properties prop, String file);
  public void writeModel(Properties prop){
    writeModel(prop, DEFAULT_MODEL_FILE);
  }
  public abstract double predict(List<Double> features);

  protected abstract void readModel(String model);

  private static String makeRegex(String regex, int num){
    StringBuilder splitter = new StringBuilder();
    for (int i = 0; i < num; i++){
      splitter.append(regex);
    }
    return splitter.toString();
  }

  protected static List<Object> toList(String line, int dim){
    assert(dim >= 1);
    List<Object> model = new ArrayList<>();
    if (dim == 1){
      model.addAll(Arrays.stream(line.split(",")).map((Double::parseDouble)).collect(Collectors.toList()));
    }
    else { //dim > 1
      String splitter = makeRegex("]", dim - 1) + ", " + makeRegex("[", dim - 1);
      String[] elements = line.split(Pattern.quote(splitter));
      for (String elem : elements){
        model.add(new ArrayList<>(toList(elem, dim - 1)));
      }
    }
    return model;
  }

}
