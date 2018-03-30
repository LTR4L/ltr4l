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

import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Regularization;
import org.ltr4l.trainers.MLPTrainer;

import java.io.IOException;
import java.io.Reader;
import java.util.List;

public class NNMLP extends MLP {
  public NNMLP(int featureLength, NetworkShape networkShape, Optimizer.OptimizerFactory optFact, Regularization regularization, String weightModel) {
    super(featureLength, networkShape, optFact, regularization, weightModel);
  }

  public NNMLP(Reader reader, MLPTrainer.MLPConfig config) throws IOException {
    super(reader, config);
  }

  @Override
  protected void addOutputs(NetworkShape ns) {
    return; //TODO: Implement addOutputs...
  }

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
}
