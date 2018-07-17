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

import java.util.List;
import java.util.Random;

import org.ltr4l.nn.*;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.tools.*;

/**
 * The implementation of MLPTrainer which uses the SortNet algorithm.
 * This network trains an MLP network.
 *
 * L. Rigutini, T. Papini, M. Maggini, and F. Scarselli: SortNet: Learning to Rank by a Neural
 * Preference Function . IEEE Transactions on Neural Networks, 22, pp. 1368–1380, 2011.
 *
 */
public class SortNetTrainer extends AbstractTrainer<SortNetMLP, MLPTrainer.MLPConfig> {
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;
  protected final double[][] targets;

  SortNetTrainer(List<Query> training, List<Query> validation, MLPTrainer.MLPConfig config, SortNetMLP ranker) {
    super(training,
        validation,
        config,
        ranker,
        StandardError.SQUARE,
        new PairwiseLossCalc.SortNetLossCalc(training, validation, StandardError.SQUARE, new double[][]{{1d,0d},{0d,1}}));

    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
    targets = new double[][]{{1, 0}, {0, 1}};
  }

  SortNetTrainer(List<Query> training, List<Query> validation, MLPTrainer.MLPConfig config){
    this(training, validation, config, new SortNetMLP(training.get(0).getFeatureLength(), config));
  }

  //The following implementation is used for speed up.
  @Override
  public void train() {
    double threshold = 0.5;
    int numTrained = 0;
    for (Query query : trainingSet) {
      List<Document> docList = query.getDocList();
      for (int i = 0; i < docList.size(); i++) {
        Document doc1 = docList.get(i);
        Document doc2 = docList.get(new Random().nextInt(docList.size()));
        //If the same document is chosen at random,
        // keep choosing until a different doc is chosen.
        while (doc1 == doc2)
          doc2 = docList.get(new Random().nextInt(docList.size()));

        double delta = doc1.getLabel() - doc2.getLabel();
        if (delta == 0) { //if the label is the same, skip.

          continue;
        }
        double prediction = ranker.predict(doc1, doc2);
        if (delta * prediction < threshold) {
          if (delta > 0)
            ranker.backProp(errorFunc, targets[0]);
          else
            ranker.backProp(errorFunc, targets[1]);
          numTrained++;
          if (batchSize == 0 || numTrained % batchSize == 0) ranker.updateWeights(lrRate, rgRate);
        }
      }
    }
    if (batchSize != 0) ranker.updateWeights(lrRate, rgRate);
  }
}

