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

package org.ltr4l.nn;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.List;
import java.util.stream.Collectors;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.Ranker;
import org.ltr4l.query.Document;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;
import org.ltr4l.trainers.MLPTrainer;

public abstract class AbstractMLPBase <N extends AbstractNode, E extends AbstractEdge> extends Ranker<MLPTrainer.MLPConfig> implements MLPInterface {

  protected final List<List<N>> network;
  protected long iter;
  protected int numAccumulatedDer;
  protected final Regularization regularization;
  protected final WeightInitializer weightInit;

  /**
   * The network is constructed within the constructor of MLP.
   * Bias edges are created for each node, which will add some constant to the total input to the node.
   * The weights held by these edges are initialized with constants (regardless of weight initialization strategy).
   * @param inputDim       The number of nodes in the input layer; the dimension of the feature space.
   * @param networkShape    Contains information about the number of hidden layers, the number of nodes in each layer,
   *                        and the activation of the nodes in the layer.
   * @param optFact         Contains information about which optimizer to use for weight updating.
   * @param regularization  Contains information about what regularization to use for weight updating.
   * @param weightModel     How to initialize weights (i.e. randomly, gaussian, etc...)
   */
  public AbstractMLPBase(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    //Network shape describes number of nodes and their activation. Example:
    //[
    //[12, Sigmoid ]
    //[4 , Softmax ]
    //[1 , Identity]
    //]
    iter = 1;
    numAccumulatedDer = 0;
    this.regularization = regularization;
    addOutputs(networkShape);
    int nWeights = inputDim * networkShape.getLayerSetting(0).getNum();  //Number of weights used for Xavier initialization.
    for (int i = 1; i < networkShape.size(); i++) {
      nWeights += networkShape.getLayerSetting(i - 1).getNum() * networkShape.getLayerSetting(i).getNum();
    }
    weightInit = WeightInitializer.get(weightModel, nWeights); //must be initialized before constructing network!
    network = constructNetwork(inputDim, networkShape, optFact);
  }

  public AbstractMLPBase(int inputDim, MLPTrainer.MLPConfig config){
    this(inputDim, config.getNetworkShape(), config.getOptFact(), config.getReguFunction(), config.getWeightInit());
  }

  public AbstractMLPBase(Reader reader, MLPTrainer.MLPConfig config){
    this.network = readModel(reader);
    //The following assignments are unnecessary for prediction and ranking, if reading a model.
    iter = 1;
    numAccumulatedDer = 0;
    regularization = config.getReguFunction();
    weightInit = null;
  }

  public double[] getOutputs(){
    return getLayer(network.size() - 1).stream().mapToDouble(node -> node.getOutput()).toArray();
  }
  protected abstract List<List<N>> constructNetwork(int inputDim, NetworkShape networkShape, Optimizer.OptimizerFactory optFact);
  protected abstract void addOutputs(NetworkShape networkShape);
  protected List<N> getLayer(int i){
    return network.get(i);
  }
  protected N getNode(int i, int j){
    return network.get(i).get(j);
  }
  protected abstract List<List<N>> readModel(Reader reader);

  /**
   * Weights are held by the edges. Lists of edges are stored in nodes.
   * Since the network is a 2 dimensional list of nodes (i.e. using network.get(i) gives you layer i),
   * obtainWeights obtains the weights from the network, in order, and keeps information about the order.
   * The dimensions of the list are:
   * 1. layer number
   * 2. node number within layer
   * 3. outputEdge number within node
   * @return List of weights.
   */
  protected List<List<List<Double>>> obtainWeights(){
    return network.stream().filter(layer -> !layer.get(0).getInputEdges().isEmpty())
        .map(layer -> layer.stream()
            .map(node -> ((List<E>) node.getInputEdges()).stream()
                .map(edge -> edge.getWeight())
                .collect(Collectors.toList()))
            .collect(Collectors.toList()))
        .collect(Collectors.toList());
  }

  @Override
  public void writeModel(MLPTrainer.MLPConfig config, Writer writer) throws IOException {
    SavedModel savedModel = new SavedModel(config, obtainWeights());
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, savedModel);
  }

  @Override
  public double predict(List<Double> features){
    return forwardProp(features);
  }

  /**
   * Input each feature into the input layer of the network, and propagate forward to the last layer.
   * @param features
   * @return
   */
  @Override
  public double forwardProp(List<Double> features) {
    //Feed document features into input layer.
    List<N> layer = network.get(0); //First layer
    for (int i = 0; i < layer.size(); i++) {
      N node = layer.get(i);
      node.setOutput(features.get(i));
    }

    //Go through the rest of the layers and update the output.
    for (int layerId = 1; layerId < network.size(); layerId++) {
      layer = network.get(layerId);
      for (N node : layer) {
        node.updateOutput();
      }
    }
    return layer.get(0).getOutput(); //After the loop, layer = lastLayer
  }

  //Overloaded to accept Document object as well.
  public double forwardProp(Document doc) {
    List<Double> features = doc.getFeatures();
    return forwardProp(features);
  }

  protected void setOutputLayerDerivatives(Error errorFunc, double... targets){
    assert(targets.length == network.get(network.size() - 1).size());
    //Set the derivative for all nodes in the output layer.
    List<N> outputLayer = network.get(network.size() - 1);
    for (int i = 0; i < outputLayer.size(); i++) {
      N outputNode = outputLayer.get(i);
      double output = outputNode.getOutput();
      double der = errorFunc.der(output, targets[i]);
      outputNode.setOutputDer(der);
    }
  }

  protected static class SavedModel {

    public MLPTrainer.MLPConfig config;
    public List<List<List<Double>>> weights;

    SavedModel(){  // this is needed for Jackson...
    }

    SavedModel(MLPTrainer.MLPConfig config, List<List<List<Double>>> weights){
      this.config = config;
      this.weights = weights;
    }

    @JsonIgnore
    public List<List<Double>> getLayer(int i){
      return weights.get(i);
    }

    @JsonIgnore
    public List<Double> getNode(int i, int j){
      return weights.get(i).get(j);
    }

    @JsonIgnore
    public double getWeight(int i, int j, int k){
      return weights.get(i).get(j).get(k);
    }
  }
}
