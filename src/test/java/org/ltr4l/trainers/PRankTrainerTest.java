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

import java.io.StringReader;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.Ranker;
import org.ltr4l.evaluation.RankEval;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.RandomDataGenerator;

public class PRankTrainerTest {

  private static final String JSON_CONFIG = "{\n" +
      "  \"algorithm\" : \"PRank\",\n" +
      "  \"numIterations\" : 100,\n" +
      "  \"verbose\": false,\n" +
      "\n" +
      "  \"model\" : {\n" +
      "    \"format\" : \"json\",\n" +
      "    \"file\" : \"model/prank-model.json\"\n" +
      "  },\n" +
      "\n" +
      "  \"evaluation\" : {\n" +
      "    \"evaluator\" : \"NDCG\",\n" +
      "    \"params\" : {\n" +
      "      \"k\" : 3\n" +
      "    }\n" +
      "  }\n" +
      "}\n";

  @Test
  public void testD1S2() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(1, 2);

    QuerySet trainSet = rdg.getRandomQuerySet(2, 10, 2);
    QuerySet validSet = rdg.getRandomQuerySet(2, 10, 2);

    AbstractTrainer trainer = AbstractTrainer.TrainerFactory.getTrainer("prank", trainSet, validSet,
        new StringReader(JSON_CONFIG), null);
    trainer.trainAndValidate();

    QuerySet testSet = rdg.getRandomQuerySet(2, 100, 20);
    Ranker ranker = trainer.getRanker();
    RankEval rankEval = RankEval.RankEvalFactory.get("ndcg");
    double eval = rankEval.calculateAvgAllQueries(ranker, testSet.getQueries(), 3);
    Assert.assertTrue(eval > 0.8);
  }

  @Test
  public void testD1S3() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(1, 3);

    QuerySet trainSet = rdg.getRandomQuerySet(2, 10, 2);
    QuerySet validSet = rdg.getRandomQuerySet(2, 10, 2);

    AbstractTrainer trainer = AbstractTrainer.TrainerFactory.getTrainer("prank", trainSet, validSet,
        new StringReader(JSON_CONFIG), null);
    trainer.trainAndValidate();

    QuerySet testSet = rdg.getRandomQuerySet(2, 100, 20);
    Ranker ranker = trainer.getRanker();
    RankEval rankEval = RankEval.RankEvalFactory.get("ndcg");
    double eval = rankEval.calculateAvgAllQueries(ranker, testSet.getQueries(), 3);
    Assert.assertTrue(eval > 0.8);
  }

  @Test
  public void testD2S2() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(2, 2);

    QuerySet trainSet = rdg.getRandomQuerySet(2, 10, 2);
    QuerySet validSet = rdg.getRandomQuerySet(2, 10, 2);

    AbstractTrainer trainer = AbstractTrainer.TrainerFactory.getTrainer("prank", trainSet, validSet,
        new StringReader(JSON_CONFIG), null);
    trainer.trainAndValidate();

    QuerySet testSet = rdg.getRandomQuerySet(2, 100, 20);
    Ranker ranker = trainer.getRanker();
    RankEval rankEval = RankEval.RankEvalFactory.get("ndcg");
    double eval = rankEval.calculateAvgAllQueries(ranker, testSet.getQueries(), 3);
    Assert.assertTrue(eval > 0.8);
  }

  @Test
  public void testD2S3() throws Exception {
    RandomDataGenerator rdg = new RandomDataGenerator(2, 3);

    QuerySet trainSet = rdg.getRandomQuerySet(2, 10, 2);
    QuerySet validSet = rdg.getRandomQuerySet(2, 10, 2);

    AbstractTrainer trainer = AbstractTrainer.TrainerFactory.getTrainer("prank", trainSet, validSet,
        new StringReader(JSON_CONFIG), null);
    trainer.trainAndValidate();

    QuerySet testSet = rdg.getRandomQuerySet(2, 100, 20);
    Ranker ranker = trainer.getRanker();
    RankEval rankEval = RankEval.RankEvalFactory.get("ndcg");
    double eval = rankEval.calculateAvgAllQueries(ranker, testSet.getQueries(), 3);
    Assert.assertTrue(eval > 0.8);
  }
}
