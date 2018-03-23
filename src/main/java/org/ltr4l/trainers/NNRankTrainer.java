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

package org.ltr4l.trainers;

import org.ltr4l.nn.Activation;
import org.ltr4l.nn.MLP;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Regularization;

import java.util.List;

/**
 * The implementation of MLPTrainer which uses the
 * Neural Network Ranking (NNRank) algorithm.
 *
 */
public class NNRankTrainer extends MLPTrainer<MLP> {
  private final int outputNodeNumber;

  //Last layer of the network has a number of nodes equal to the number of categories.
  //That layer is created in the constructor, so it is not necessary to specify last layer in config file.
  NNRankTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config, true);
    outputNodeNumber = QuerySet.findMaxLabel(trainingSet) + 1;
  }

  @Override
  protected MLP constructRanker(){
    int featureLength = trainingSet.get(0).getFeatureLength();
    //Add an output layer with number of nodes equal to number of classes/relevance categories.
    NetworkShape networkShape = config.getNetworkShape();
    int outputNodeNumber = QuerySet.findMaxLabel(trainingSet);
    networkShape.add(outputNodeNumber + 1, new Activation.Sigmoid());
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new MLP(featureLength, networkShape, optFact, regularization, weightModel) {
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

  }

  private double[] targetLabel(int label) {
    double[] targets = new double[outputNodeNumber]; //initialized with 0.
    for (int index = 0; index <= label; index++)
      targets[index] = 1;
    return targets;
  }

  @Override
  public void train() {
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        int output = (int) ranker.predict(doc.getFeatures());
        int label = doc.getLabel();
        if (output != label) {
          ranker.backProp(errorFunc, targetLabel(label));
          ranker.updateWeights(lrRate, rgRate);
        }
      }
    }
  }
}
