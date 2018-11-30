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
import java.util.*;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.ltr4l.boosting.AdaBoost;
import org.ltr4l.boosting.Ensemble;
import org.ltr4l.boosting.RankBoost;
import org.ltr4l.nn.*;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
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

  public List<Document> sort(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    ranks.sort((docA, docB) -> Double.compare(predict(docB.getFeatures()), predict(docA.getFeatures()))); //reversed for high to low.
    return ranks;
  }

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
    public static Ranker get(Reader reader, Config override, int featLength, int maxLabel) {
      String algorithm;
      ObjectMapper mapper = new ObjectMapper();
      mapper.disable(JsonParser.Feature.AUTO_CLOSE_SOURCE);
      try {
        Map model = mapper.readValue(reader, Map.class);
        algorithm = ((String)model.get("algorithm")).toLowerCase();
        reader.reset();
      } catch (IOException ioe) {
        throw new RuntimeException(ioe);
      }
      return get(algorithm, reader, override, featLength, maxLabel);
    }

    public static Ranker get(String algorithm, Reader reader, Config override, int featLength, int maxLabel){
      assert(featLength > 0 && maxLabel > 0);
      String alg = algorithm.toLowerCase();
      switch (alg){
        case "prank":
          return new PRank(featLength, maxLabel);
        case "oap":
          OAPBPMConfig config = Config.getConfig(reader, Config.ConfigType.OAP);
          config.overrideBy(override);
          int pNum = config.getPNum();
          double bernNum = config.getBernNum();
          return new OAPBPMRank(featLength, maxLabel, pNum, bernNum);
        case "nnrank":
          MLPTrainer.MLPConfig mlpConfig = Config.getConfig(reader, Config.ConfigType.MLP);
          mlpConfig.overrideBy(override);
          NetworkShape networkShape = mlpConfig.getNetworkShape();
          networkShape.add(maxLabel + 1, Activation.Type.Sigmoid);
          return new NNMLP(featLength, networkShape, mlpConfig.getOptFact(), mlpConfig.getReguFunction(), mlpConfig.getWeightInit());
        // The algorithms below do not require max label, however they can still be specified.
        default:
          return get(algorithm, reader, override, featLength);
      }

    }

    public static Ranker get(Reader reader, Config override, int featLength) {
      String algorithm;
      ObjectMapper mapper = new ObjectMapper();
      mapper.disable(JsonParser.Feature.AUTO_CLOSE_SOURCE);
      try {
        Map model = mapper.readValue(reader, Map.class);
        algorithm = ((String)model.get("algorithm")).toLowerCase();
        reader.reset();
      } catch (IOException ioe) {
        throw new RuntimeException(ioe);
      }
      return get(algorithm, reader, override, featLength);
    }

    public static Ranker get(String algorithm, Reader reader, Config override, int featLength) {
      assert(featLength > 0);
      String alg = algorithm.toLowerCase();
      MLPTrainer.MLPConfig config = Config.getConfig(reader, Config.ConfigType.MLP);
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

    public static Ranker getFromModel(Reader reader) throws IOException{
      if (reader instanceof FileReader)
        throw new IllegalArgumentException("FileReader is currently unsupported...");
      BufferedReader br;
      //Here a check for markSupported() is not needed, as BufferedReader supports it.
      //BufferedReader is needed for readLine(), as model file is likely to exceed
      //reasonable readAheadLimits; thus ObjectMapper should not be used...
      if (reader instanceof BufferedReader)
        br = (BufferedReader) reader;
      else
        br = new BufferedReader(reader);
      br.mark(8192);
      String line;
      String algorithm = null;
      while ((line = br.readLine()) != null) {
        line = line.trim();
        if (line.startsWith("\"algorithm\" :")) {
          line = line.split(":")[1].trim(); //Note: no check for extra colons...
          assert(line.startsWith("\"") && line.endsWith("\""));
          algorithm = line.substring(1, line.length() - 2).toLowerCase();
          //System.out.println("Algorithm is " + algorithm);
          break;
        }
      }
      Objects.requireNonNull(algorithm, "Model file does not contain algorithm name!");
      br.reset();
      return getFromModel(algorithm, reader);
    }

    public static Ranker getFromModel(String algorithm, Reader reader) {
      String alg = algorithm;
      try {
        if (alg.equals("prank")) {
          return PRank.readModel(reader);
        }
        else if (alg.equals("oap")) {
          return OAPBPMRank.readModel(reader); //This returns PRank (which is fine for predicting model.)
        }

        switch (alg) {  //For MLP Rankers
          case "nnrank":
            return new NNMLP(reader);
          case "ranknet":
          case "franknet":
          case "lambdarank":
            return new RankNetMLP(reader);
          case "sortnet":
            return new SortNetMLP(reader); //TODO: add ModelReader to SortNet.
          case "listnet":
            return new ListNetMLP(reader);
          case "lambdamart":
            return new Ensemble(reader);
          case "rankboost":
            return new RankBoost(reader);
          case "adaboost":
            return new AdaBoost(reader);
          default:
            throw new IllegalArgumentException("Specified algorithm does not exist.");
        }
      }
      catch (IOException e) {
        throw new RuntimeException(e);
      }
    }
  }
}
