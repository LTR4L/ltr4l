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

package org.ltr4l.cli;

import org.apache.commons.cli.CommandLine;
import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.tools.Config;

public class TrainTest {

  @Test
  public void testGetConfigPath() throws Exception {
    CommandLine line = Train.getCommandLine(Train.createOptions(), new String[]{"franknet"});
    Assert.assertEquals("confs/franknet.json", Train.getConfigPath(line, line.getArgs()));
  }

  @Test
  public void testGetConfigPathSpecified() throws Exception {
    CommandLine line = Train.getCommandLine(Train.createOptions(), new String[]{"franknet", "-config", "myconfig.json"});
    Assert.assertEquals("myconfig.json", Train.getConfigPath(line, line.getArgs()));
  }

  @Test
  public void testOptionalConfigNoOverride() throws Exception {
    CommandLine line = Train.getCommandLine(Train.createOptions(),
        new String[]{"franknet"});
    String configPath = Train.getConfigPath(line, line.getArgs());
    Config config = Train.createOptionalConfig(configPath, line);
    Assert.assertEquals(100, config.numIterations);
    Assert.assertEquals("data/MQ2008/Fold1/train.txt", config.dataSet.training);
    Assert.assertEquals("data/MQ2008/Fold1/vali.txt", config.dataSet.validation);
    Assert.assertEquals("model/franknet-model.json", config.model.file);
    Assert.assertEquals("report/franknet-report.csv", config.report.file);
  }

  @Test
  public void testOptionalConfigOverride() throws Exception {
    CommandLine line = Train.getCommandLine(Train.createOptions(),
        new String[]{"franknet",
            "-iterations", "500",
            "-training", "mytrain.txt",
            "-validation", "myvali.txt",
            "-model", "mymodel.json",
            "-nomodel",
            "-report", "myreport.csv"});
    String configPath = Train.getConfigPath(line, line.getArgs());
    Config config = Train.createOptionalConfig(configPath, line);
    Assert.assertEquals(500, config.numIterations);
    Assert.assertEquals("mytrain.txt", config.dataSet.training);
    Assert.assertEquals("myvali.txt", config.dataSet.validation);
    Assert.assertEquals("mymodel.json", config.model.file);
    Assert.assertTrue(config.nomodel);
    Assert.assertEquals("myreport.csv", config.report.file);
  }
}
