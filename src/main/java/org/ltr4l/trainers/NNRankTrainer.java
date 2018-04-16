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

import java.io.Reader;

import org.ltr4l.nn.*;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Regularization;

/**
 * The implementation of MLPTrainer which uses the
 * Neural Network Ranking (NNRank) algorithm.
 *
 */
public class NNRankTrainer extends MLPTrainer<MLP> {
  private final int outputNodeNumber;
  private final double[][] targets;

  //Last layer of the network has a number of nodes equal to the number of categories.
  //That layer is created in the constructor, so it is not necessary to specify last layer in config file.
  NNRankTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
    super(training, validation, reader, override, true);
    outputNodeNumber = QuerySet.findMaxLabel(trainingSet);
    targets = new double[outputNodeNumber + 1][outputNodeNumber]; //For 2 output nodes, there should be 3 targets
    for(int i = 0; i < outputNodeNumber + 1; i++)
      targets[i] = makeTarget(i, outputNodeNumber);
  }

  @Override
  protected NNMLP constructRanker(){
    int featureLength = trainingSet.get(0).getFeatureLength();
    //Add an output layer with number of nodes equal to number of classes/relevance categories.
    NetworkShape networkShape = config.getNetworkShape();
    int outputNodeNumber = QuerySet.findMaxLabel(trainingSet);
    networkShape.add(outputNodeNumber, new Activation.Sigmoid());
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new NNMLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  @Override
  public void train() {
    int numTrained = 0;
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        int output = (int) ranker.predict(doc.getFeatures());
        int label = doc.getLabel();
        if (output != label) {
          ranker.backProp(errorFunc, targets[label]);
          numTrained++;
          if (batchSize == 0 || numTrained % batchSize == 0) ranker.updateWeights(lrRate, rgRate);
        }
      }
    }
    if (batchSize != 0) ranker.updateWeights(lrRate, rgRate);
  }

  private static double[] makeTarget(int label, int outputNodeNumber){
    double[] targets = new double[outputNodeNumber]; //initialized with 0.
    for (int index = 0; index < label; index++)
      targets[index] = 1;
    return targets;
  }

}
