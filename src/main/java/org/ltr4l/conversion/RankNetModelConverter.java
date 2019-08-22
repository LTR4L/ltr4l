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
package org.ltr4l.conversion;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.nn.AbstractMLPBase;
import org.ltr4l.nn.Activation;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class RankNetModelConverter implements LTRModelConverter {

  @Override
  public void write(SolrLTRModel model, Writer writer) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, model);
  }

  @Override
  public SolrLTRModel convert(Reader reader, List<String> features) {
    SolrLTRModel solrModel = new SolrLTRModel();
    solrModel.clazz = "org.apache.solr.ltr.model.NeuralNetworkModel";
    //TODO: hardcoded...
    solrModel.name = "rankNetModel";
    solrModel.store = "ltrFeatureStore";
    List<SolrLTRModel.Feature> features1 = features.stream()
        .map(SolrLTRModel.Feature::new)
        .collect(Collectors.toCollection(ArrayList::new));
    solrModel.features = features1;
    solrModel.params = new HashMap<>();
    List<Layer> layers = new ArrayList<>();
    solrModel.params.put("layers", layers);
    try {
      ObjectMapper mapper = new ObjectMapper();
      AbstractMLPBase.SavedModel ltr4lNNModel = mapper.readValue(reader, AbstractMLPBase.SavedModel.class);
      assert(ltr4lNNModel.config.algorithm.equals("ranknet"));
      int numLayers = ltr4lNNModel.weights.size();
      List<Map<String, Object>> layersSetting = (List<Map<String, Object>>)ltr4lNNModel.config.params.get("layers");
      for (int i = 0; i < numLayers; i++) {
        String activation;
        if (i == numLayers - 1)
          activation = "sigmoid";
        else {
          Map<String, Object> layerParams = layersSetting.get(i);
          activation = ((String) layerParams.get("activator")).toLowerCase();
        }
        layers.add(convertLayer(ltr4lNNModel.getLayer(i), activation));
      }
      return solrModel;
    } catch (Exception e) {
      throw new RuntimeException(e); //TODO: Better error handling
    }
  }

  private Layer convertLayer(List<List<Double>> ltr4lLayer, String activation) {
    //Extract biases
    List<Double> biases = new ArrayList<>();
    List<List<Double>> matrix = new ArrayList<>();
    for(List<Double> nodeEdges : ltr4lLayer) {
      biases.add(nodeEdges.get(0));
      for (int i = 1; i < nodeEdges.size(); i++) {
        matrix.add(nodeEdges.subList(1, nodeEdges.size()));
      }
    }
    Layer solrLayer = new Layer();
    solrLayer.activation = activation;
    solrLayer.biases = biases;
    solrLayer.matrix = matrix;
    return solrLayer;
  }


  public static class Layer {
    public List<List<Double>> matrix;
    public List<Double> biases;
    public String activation;
  }


}
