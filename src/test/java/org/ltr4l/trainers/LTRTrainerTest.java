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

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;
import java.util.List;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.Ranker;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.*;

public class LTRTrainerTest {

  @Test
  public void testGetKofNDCGatK() throws Exception {
    final String JSON1 = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100,\n" +
        "\n" +
        "  \"evaluation\" : {\n" +
        "    \"evaluator\" : \"NDCG\",\n" +
        "    \"params\" : {\n" +
        "      \"k\" : 7\n" +
        "    }\n" +
        "  }\n" +
        "}\n";

    LTRTrainer trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON1));
    Assert.assertEquals(7, trainer.ndcgK);
  }

  @Test
  public void testDefaultKofNDCGatK() throws Exception {
    final String JSON1 = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100\n" +
        "}\n";

    LTRTrainer trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON1));
    Assert.assertEquals(10, trainer.ndcgK);
  }

  private static class NullRanker extends Ranker<Config> {

    @Override
    public void writeModel(Config config, Writer writer) throws IOException {

    }

    @Override
    public double predict(List<Double> features) {
      return 0;
    }
  }

  private static class NullLTRTrainer extends LTRTrainer<NullRanker, Config> {

    NullLTRTrainer(QuerySet training, QuerySet validation, Reader reader) {
      super(training, validation, reader);
    }

    @Override
    double calculateLoss(List<Query> queries) {
      return 0;
    }

    @Override
    protected org.ltr4l.tools.Error makeErrorFunc() {
      return null;
    }

    @Override
    protected <R extends Ranker> R constructRanker() {
      return null;
    }

    @Override
    public Class<Config> getConfigClass() {
      return Config.class;
    }

    @Override
    public void train() {

    }
  }
}
