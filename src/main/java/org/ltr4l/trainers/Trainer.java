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

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.List;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;


public interface Trainer {

  void train();

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
     * @param configFile The Config file containing parameters needed for Ranker class.
     * @return new class which implements trainer.
     */
    public static Trainer getTrainer(String algorithm, QuerySet trainingSet, QuerySet validationSet, String configFile, Config override) {
      try(Reader reader = new FileReader(configFile)){
        switch (algorithm.toLowerCase()) {
          case "prank":
            return new PRankTrainer(trainingSet, validationSet, reader, override);
          case "oap":
            return new OAPBPMTrainer(trainingSet, validationSet, reader, override);
          case "ranknet":
            return new RankNetTrainer(trainingSet, validationSet, reader, override);
          case "franknet":
            return new FRankTrainer(trainingSet, validationSet, reader, override);
          case "lambdarank":
            return new LambdaRankTrainer(trainingSet, validationSet, reader, override);
          case "nnrank":
            return new NNRankTrainer(trainingSet, validationSet, reader, override);
          case "sortnet":
            return new SortNetTrainer(trainingSet, validationSet, reader, override);
          case "listnet":
            return new ListNetTrainer(trainingSet, validationSet, reader, override);
          default:
            return null;
        }
      }
      catch (IOException e){
        throw new IllegalArgumentException(e);
      }
    }
  }
}

