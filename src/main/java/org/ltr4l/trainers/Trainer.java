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

import org.ltr4l.tools.Config;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.List;


public interface Trainer {

  void train();

  //default NDCG@10
  default void validate(int iter) {
    validate(iter, 10);

  }

  void validate(int iter, int pos);

  double[] calculateLoss();

  List<Document> sortP(Query query);

  void trainAndValidate();

  class TrainerFactory {

    /**
     * This returns the appropriate implementation of Trainer depending on the algorithm.
     * @param algorithm Algorithm/implementation to be used.
     * @param trainingSet The QuerySet containing the data to be used for training.
     * @param validationSet The QuerySet containing the data to be used for validation.
     * @param config The Config class containing parameters needed for Ranker class.
     * @return new class which implements trainer.
     */
    public static Trainer getTrainer(String algorithm, QuerySet trainingSet, QuerySet validationSet, Config config) {
      switch (algorithm.toLowerCase()) {
        case "prank":
          return new PRankTrainer(trainingSet, validationSet, config);
        case "oap":
          return new OAPBPMTrainer(trainingSet, validationSet, config);
        case "ranknet":
          return new RankNetTrainer(trainingSet, validationSet, config);
        case "franknet":
          return new FRankTrainer(trainingSet, validationSet, config);
        case "lambdarank":
          return new LambdaRankTrainer(trainingSet, validationSet, config);
        case "nnrank":
          return new NNRankTrainer(trainingSet, validationSet, config);
        case "sortnet":
          return new SortNetTrainer(trainingSet, validationSet, config);
        case "listnet":
          return new ListNetTrainer(trainingSet, validationSet, config);
        default:
          return null;
      }
    }
  }
}

