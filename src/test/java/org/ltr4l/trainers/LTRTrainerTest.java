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

import java.io.File;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.io.Writer;
import java.util.List;

import org.junit.After;
import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.Ranker;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.*;

public class LTRTrainerTest {

  private LTRTrainer trainer;

  @After
  public void tearDown(){
    Report report = trainer.report;
    if(report != null){
      report.close();
      if(report.getReportFile() != null){
        File reportFile = new File(report.getReportFile());
        reportFile.delete();
      }
    }
  }

  @Test
  public void testGetKofNDCGatK() throws Exception {
    final String JSON = "{\n" +
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

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals(7, trainer.evalK);
  }

  @Test
  public void testDefaultKofNDCGatK() throws Exception {
    final String JSON = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100\n" +
        "}\n";

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals(10, trainer.evalK);
  }

  @Test
  public void testGetModelFile() throws Exception {
    final String JSON = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100,\n" +
        "\n" +
        "  \"model\" : {\n" +
        "    \"format\" : \"json\",\n" +
        "    \"file\" : \"model/franknet-model.json\"\n" +
        "  }\n" +
        "}\n";

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals("model/franknet-model.json", trainer.modelFile);
  }

  @Test
  public void testDefaultModelFile() throws Exception {
    final String JSON = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100\n" +
        "}\n";

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals("model/model.txt", trainer.modelFile);
  }

  @Test
  public void testGetReportFile() throws Exception {
    final String JSON = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100,\n" +
        "\n" +
        "  \"report\" : {\n" +
        "    \"format\" : \"csv\",\n" +
        "    \"file\" : \"report/franknet-report.csv\"\n" +
        "  }\n" +
        "}\n";

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals("report/franknet-report.csv", trainer.report.getReportFile());
  }

  @Test
  public void testDefaultReportFile() throws Exception {
    final String JSON = "{\n" +
        "  \"algorithm\" : \"FRankNet\",\n" +
        "  \"numIterations\" : 100\n" +
        "}\n";

    trainer = new NullLTRTrainer(new QuerySet(), new QuerySet(), new StringReader(JSON), null);
    Assert.assertEquals("report/report.csv", trainer.report.getReportFile());
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

    NullLTRTrainer(QuerySet training, QuerySet validation, Reader reader, Config override) {
      super(training, validation, reader, override);
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
