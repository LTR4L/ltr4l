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
    
    //For rankers which need information about the max label (needed for structure of network)
    public static Ranker get(String algorithm, String configFile, Config override, int featLength, int maxLabel){
      assert(featLength > 0 && maxLabel > 0);
      try (Reader reader = new FileReader(configFile)) {
        String alg = algorithm.toLowerCase();

        switch (alg){
          case "prank":
            return new PRank(featLength, maxLabel);
          case "oap":
            OAPBPMConfig config = getConfig(reader, OAPBPMTrainer.OAPBPMConfig.class); //TODO: hardcoded...
            config.overrideBy(override);
            int pNum = config.getPNum();
            double bernNum = config.getBernNum();
            return new OAPBPMRank(featLength, maxLabel, pNum, bernNum);
          case "nnrank":
            MLPTrainer.MLPConfig mlpConfig = getConfig(reader, MLPTrainer.MLPConfig.class);
            mlpConfig.overrideBy(override);
            NetworkShape networkShape = mlpConfig.getNetworkShape();
            networkShape.add(maxLabel + 1, new Activation.Sigmoid());
            return new NNMLP(featLength, networkShape, mlpConfig.getOptFact(), mlpConfig.getReguFunction(), mlpConfig.getWeightInit());
          // The algorithms below do not require max label, however they can still be specified.
          default:
            return get(algorithm, configFile, override, featLength);
        }
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public static Ranker get(String algorithm, String configFile, Config override, int featLength) {
      assert(featLength > 0);
      try (Reader reader = new FileReader(configFile)) {
        String alg = algorithm.toLowerCase();
        MLPTrainer.MLPConfig config = getConfig(reader, MLPTrainer.MLPConfig.class);
        config.overrideBy(override);
        switch (alg) {
          case "prank":
          case "oap":
          case "nnrank":
            throw new IllegalArgumentException("Must must specify max label in dataset! Use get(String algorithm, String configFile, Config override, int featLength, int maxLabel)");
          case "ranknet":
          case "franknet":
          case "lambdarank":
            return new RankNetMLP(featLength, config);
          case "sortnet":
            return new SortNetMLP(featLength, config);
          case "listnet":
            return new ListNetMLP(featLength, config);
          default:
            throw new IllegalArgumentException("Specified algorithm does not exist.");
        }
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    }

    public static Ranker getFromModel(String algorithm, String configFile, Config override) {
      try (Reader reader = new FileReader(configFile)) {
        String alg = algorithm.toLowerCase();
        Config config = getConfig(reader, Config.class);

        if ((config.model == null || config.model.file == null || config.model.file.isEmpty()))
          throw new IllegalArgumentException("No model specified");

        if (alg.equals("prank")) {
          return PRank.readModel(reader);
        }
        else if (alg.equals("oap")) {
          return OAPBPMRank.readModel(reader); //This returns PRank (which is fine for predicting model.)
        }

        MLPTrainer.MLPConfig mlpConfig = getConfig(reader, MLPTrainer.MLPConfig.class); //TODO: Don't make new config...
        mlpConfig.overrideBy(override);
        switch (alg) {  //For MLP Rankers
          case "nnrank":
            return new NNMLP(reader, mlpConfig);
          case "ranknet":
          case "franknet":
          case "lambdarank":
            return new RankNetMLP(reader, mlpConfig);
          case "sortnet":
            return new SortNetMLP(reader, mlpConfig); //TODO: add ModelReader to SortNet.
          case "listnet":
            return new ListNetMLP(reader, mlpConfig);
          default:
            throw new IllegalArgumentException("Specified algorithm does not exist.");
        }
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }

    //TODO: Move elsewhere? Same as LTRTrainer.getConfig
    protected static <C extends Config> C getConfig(Reader reader, Class<C> configClass){
      ObjectMapper mapper = new ObjectMapper();
      try {
        return mapper.readValue(reader, configClass);
      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
}
