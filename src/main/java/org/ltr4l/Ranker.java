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
import org.ltr4l.tools.Regularization;
import org.ltr4l.trainers.MLPTrainer;
import org.ltr4l.trainers.OAPBPMTrainer;
import org.ltr4l.trainers.PRankTrainer;

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
          int maxLabel = QuerySet.findMaxLabel(testSet.getQueries());
          return PRankTrainer.getPRank(featLength, maxLabel);
        } else if (alg.equals("oap")) { //TODO: Return PRank instead?
          int maxLabel = QuerySet.findMaxLabel(testSet.getQueries());
          OAPBPMTrainer.OAPBPMConfig config = getConfig(reader, OAPBPMTrainer.OAPBPMConfig.class);
          config.overrideBy(override);
          int pNum = config.getPNum();
          double bernNum = config.getBernNum();
          //int pNum = Config.getInt(config.params, "N", 1);
          //double bernNum = Config.getDouble(config.params, "bernoulli", 0.03);
          return OAPBPMTrainer.getOAP(featLength, maxLabel, pNum, bernNum);
        }
        MLPTrainer.MLPConfig config = getConfig(reader, MLPTrainer.MLPConfig.class);
        config.overrideBy(override);
        AbstractMLP ranker;
        switch (alg) {  //For MLP Rankers
          case "nnrank":
            NetworkShape networkShape = config.getNetworkShape();
            int outputNodeNumber = QuerySet.findMaxLabel(testSet.getQueries());
            networkShape.add(outputNodeNumber + 1, new Activation.Sigmoid());
            return new MLP(featLength, config) {
              @Override
              public double predict(List<Double> features) {
                double threshold = 0.5;
                forwardProp(features);
                for (int nodeId = 0; nodeId < network.get(network.size() - 1).size(); nodeId++) {
                  MNode node = network.get(network.size() - 1).get(nodeId);
                  if (node.getOutput() < threshold)
                    return nodeId - 1;
                }
                return network.get(network.size() - 1).size() - 1;
              }
            };
          case "ranknet":
          case "franknet":
          case "lambdarank":
            ranker = new RankNetMLP(featLength, config);
            if (config.model == null || config.model.file == null || config.model.file.isEmpty()) {
              ranker.load(reader);
            }
            return ranker;
          case "sortnet":
            ranker = new SortNetMLP(featLength, config);
            return new SortNetMLP(featLength, config);
          case "listnet":
            return new ListNetMLP(featLength, config);
          default:
            throw new IllegalArgumentException("Specified algorithm does not exist.");
        }

      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }


    }

    protected static <C extends Config> C getConfig(Reader reader, Class<C> configClass){
      ObjectMapper mapper = new ObjectMapper();
      try {
        return mapper.readValue(reader, configClass);
      } catch (IOException e) {
        throw new IllegalArgumentException(e);
      }
    }
}
