package org.ltr4l.cli;

import org.apache.commons.cli.CommandLine;
import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.tools.Config;

public class PredictTest {

  @Test
  public void testGetModelPath() throws Exception {
    CommandLine line = Predict.getCommandLine(Predict.createOptions(), new String[]{"franknet"});
    Assert.assertEquals("model/franknet-model.json", Predict.getModelPath(line, line.getArgs()));
  }

  @Test
  public void testGetModelPathSpecified() throws Exception {
    CommandLine line = Predict.getCommandLine(Predict.createOptions(), new String[]{"franknet", "-model", "mymodel.json"});
    Assert.assertEquals("mymodel.json", Predict.getModelPath(line, line.getArgs()));
  }

  @Test
  public void testOptionalConfigNoOverride() throws Exception {
    CommandLine line = Predict.getCommandLine(Predict.createOptions(),
        new String[]{"franknet",});
    String modelPath = Predict.getModelPath(line, line.getArgs());
    Config config = Predict.createOptionalConfig(modelPath, line);
    Assert.assertEquals(100, config.numIterations);
    Assert.assertEquals("data/MQ2008/Fold1/test.txt", config.dataSet.test);
    Assert.assertEquals("model/franknet-model.json", config.model.file);
    Assert.assertEquals("report/franknet-report.csv", config.report.file);
  }

  @Test
  public void testOptionalConfigOverride() throws Exception {
    CommandLine line = Predict.getCommandLine(Predict.createOptions(),
        new String[]{"franknet",
            "-test", "myTest.txt",
            "-report", "myreport.csv",
            "-eval", "myEvaluator",
            "-k", "10"});
    String modelPath = Predict.getModelPath(line, line.getArgs());
    Config config = Predict.createOptionalConfig(modelPath, line);
    Assert.assertEquals("myTest.txt", config.dataSet.test);
    Assert.assertEquals("myreport.csv", config.report.file);
    Assert.assertEquals("myEvaluator", config.evaluation.evaluator);
    Assert.assertEquals(10, (int) config.evaluation.params.get("k"));
  }

}