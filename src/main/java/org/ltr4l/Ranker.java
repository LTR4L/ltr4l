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
package org.ltr4l;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.ltr4l.nn.*;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.trainers.*;
import org.ltr4l.trainers.OAPBPMTrainer.*;
import org.ltr4l.trainers.PRankTrainer.*;


/**
 * Ranker classes use a model to make predictions for a document.
 * Models (weights, thresholds, etc...) held by Ranker classes
 * can be trained by trainers.
 *
 * Rankers are the model holders.
 */
public abstract class Ranker<C extends Config> {

  public void writeModel(C config, String file) throws IOException {
    try(Writer writer = new FileWriter(file)){
      writeModel(config, writer);
    }
  }

  public abstract void writeModel(C config, Writer writer) throws IOException;

  public abstract double predict(List<Double> features);

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

  public static class RankerFactory {
    public static Ranker get(String algorithm, QuerySet testSet, String configFile, Config override) {
      try (Reader reader = new FileReader(configFile)) {
        String alg = algorithm.toLowerCase();
        int featLength = testSet.getFeatureLength();

        if (alg.equals("prank")) {
          Config config = getConfig(reader, Config.class);
          int maxLabel = QuerySet.findMaxLabel(testSet.getQueries());
          if (!(config.model == null || config.model.file == null || config.model.file.isEmpty()))
            return PRank.readModel(reader);
          return new PRank(featLength, maxLabel);
        } else if (alg.equals("oap")) {
          int maxLabel = QuerySet.findMaxLabel(testSet.getQueries());
          OAPBPMConfig config = getConfig(reader, OAPBPMTrainer.OAPBPMConfig.class); //TODO: hardcoded...
          config.overrideBy(override);
          int pNum = config.getPNum();
          double bernNum = config.getBernNum();
          if (!(config.model == null || config.model.file == null || config.model.file.isEmpty()))
            return OAPBPMRank.readModel(reader);
          return new OAPBPMRank(featLength, maxLabel, pNum, bernNum);
        }

        MLPTrainer.MLPConfig config = getConfig(reader, MLPTrainer.MLPConfig.class);
        config.overrideBy(override);
        switch (alg) {  //For MLP Rankers
          case "nnrank":
            NetworkShape networkShape = config.getNetworkShape();
            int outputNodeNumber = QuerySet.findMaxLabel(testSet.getQueries());
            networkShape.add(outputNodeNumber + 1, new Activation.Sigmoid());
            if (!(config.model == null || config.model.file == null || config.model.file.isEmpty()))
              return new NNMLP(reader, config);
            //TODO: Implement addOutputs correctly...
            return new NNMLP(featLength, networkShape, config.getOptFact(), config.getReguFunction(), config.getWeightInit());
          case "ranknet":
          case "franknet":
          case "lambdarank":
            if (!(config.model == null || config.model.file == null || config.model.file.isEmpty())) {
              return new RankNetMLP(reader, config);
            }
            return new RankNetMLP(featLength, config);
          case "sortnet":
            return new SortNetMLP(featLength, config); //TODO: add ModelReader to SortNet.
          case "listnet":
            if (!(config.model == null || config.model.file == null || config.model.file.isEmpty()))
              return new ListNetMLP(reader, config);
            return new ListNetMLP(featLength, config);
          default:
            throw new IllegalArgumentException("Specified algorithm does not exist.");
        }
      }
      catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
  }

    protected static <C extends Config> C getConfig(Reader reader, Class<C> configClass){ //TODO: Move elsewhere?
      ObjectMapper mapper = new ObjectMapper();
      try {
        return mapper.readValue(reader, configClass);
      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
}
