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

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.ltr4l.nn.*;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.tools.Regularization;

/**
 * The implementation of MLPTrainer which uses the SortNet algorithm.
 * This network trains an MLP network.
 *
 */
public class SortNetTrainer extends LTRTrainer<SortNetMLP> {
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;
  protected final double[][] targets;

  SortNetTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
    targets = new double[][]{{1, 0}, {0, 1}};
  }

  @Override
  protected Error makeErrorFunc(){
    return new Error.Square();
  }

  @Override
  protected SortNetMLP constructRanker() {
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new SortNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  //The following implementation is used for speed up.
  @Override
  public void train() {
    double threshold = 0.5;
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
        if (delta == 0) //if the label is the same, skip.
          continue;
        double prediction = ranker.predict(doc1, doc2);
        if (delta * prediction < threshold) {
          if (delta > 0)
            ranker.backProp(errorFunc, targets[0]);
          else
            ranker.backProp(errorFunc, targets[1]);

          ranker.updateWeights(lrRate, rgRate);
        }
      }
    }
  }

  public double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      Document[][] pairs = query.orderDocPairs();
      if (pairs == null)
        continue;
      double queryLoss = 0d;
      for (Document[] pair : pairs) {
        List<Double> combinedFeatures = new ArrayList<>(pair[0].getFeatures());
        combinedFeatures.addAll(pair[1].getFeatures());
        ranker.forwardProp(combinedFeatures);
        double[] outputs = ranker.getOutputs();
        queryLoss += errorFunc.error(outputs[0], targets[0][0]);
        queryLoss += errorFunc.error(outputs[1], targets[0][1]);
      }
      loss += queryLoss / (double) pairs.length;
    }
    return loss / (double) queries.size();
  }

  @Override
  public List<Document> sortP(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    //Reverse order to go from highest to lowest instead of lowest to highest.
    ranks.sort((docA, docB) -> Double.compare(0, ranker.predict(docA, docB)));
/*        ranks.sort(new Comparator<Document>() {
            @Override
            public int compare(Document o1, Document o2) {
                double prediction = smlp.predict(o1, o2);
                if (prediction > 0) return -1; //-1 because it should not be changed; higher is earlier
                if (prediction < 0) return 1;
                return 0;
            }
        });*/
    return ranks;
  }
}

