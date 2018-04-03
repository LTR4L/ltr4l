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

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.cli.*;
import org.ltr4l.Ranker;
import org.ltr4l.Version;
import org.ltr4l.evaluation.RankEval;
import org.ltr4l.nn.MLP;
import org.ltr4l.nn.RankNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Report;
import org.ltr4l.trainers.MLPTrainer;

import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public class Predict {

  private static final String REQUIRED_ARG = "<LTR-algorithm-name>";

  public static void main (String args[]) throws Exception{
    Options options = createOptions();
    CommandLine line = getCommandLine(options, args);

    if(line.hasOption("help")) printUsage(options);

    if(line.hasOption("version")){
      System.out.printf("%s release %s\n", MethodHandles.lookup().lookupClass().getCanonicalName(), Version.version);
      System.exit(0);
    }

    // get LTR-algorithm-name
    String[] params = line.getArgs();
    if(params == null || params.length == 0){
      System.err.printf("No required argument %s specified", REQUIRED_ARG);
      printUsage(options);
    }
    else if(params.length > 1){
      System.err.printf("Too many argument is specified: %s", params[1]);
      printUsage(options);
    }

    String modelPath = getModelPath(line, params);
    Config optionalConfig = createOptionalConfig(modelPath, line);
    QuerySet testSet = QuerySet.create(optionalConfig.dataSet.test);
    Ranker ranker = getRanker(modelPath, params);

    evaluate(ranker, testSet.getQueries(), optionalConfig);

  }

  public static Options createOptions(){
    Option help = new Option( "help", "print this message" );
    Option modelFile = Option.builder("model").argName("file").hasArg()
        .desc("use given file for configuration and model").build();
    Option testDataSet = Option.builder("test").argName("file").hasArg()
        .desc("use given file for testing the model").build();
    Option reportFile = Option.builder("report").argName("file").hasArg()
        .desc("specify report file name").build();
    Option evalType = Option.builder("eval").argName("evalType").hasArg()
        .desc("specify type of evaluator").build();
    Option k = Option.builder("k").argName("k").hasArg()
        .desc("specify k-value for evaluators which use @k").build();
    Option version = new Option( "version", "print the version information and exit" );
    Option verbose = new Option( "verbose", "be extra verbose" );
    Option debug = new Option( "debug", "print debugging information" );

    Options options = new Options();
    options.addOption(help)
        .addOption(modelFile)
        .addOption(testDataSet)
        .addOption(reportFile)
        .addOption(evalType)
        .addOption(k)
        .addOption(version)
        .addOption(verbose)
        .addOption(debug);
    return options;
  }

  public static void printUsage(Options options){
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "predict " + REQUIRED_ARG,
        "\nExecute Learning-to-Rank predicting algorithm. The algorithm is specified by the required argument <LTR-algorithm-name>." +
            " The program will look for the model file \"model/<LTR-algorithm-name>-model.json\"" +
            " unless model option is specified." +
            " The following options can be specified in order to override the existing settings in the config file.\n\n",
        options, null, true );
    System.exit(0);
  }

  public static CommandLine getCommandLine(Options options, String[] args){
    CommandLineParser parser = new DefaultParser();
    CommandLine line = null;
    try {
      // parse the command line arguments
      line = parser.parse(options, args);
    }
    catch(ParseException exp) {
      // oops, something went wrong
      System.err.printf("Parsing failed. Reason: %s\n\n", exp.getMessage());
      printUsage(options);
    }

    return line;
  }

  public static String getModelPath(CommandLine line, String[] params){
    assert(params.length == 1);
    return line.hasOption("model") ? line.getOptionValue("model") : String.format("model/%s-model.json", params[0]);
  }

  public static Ranker getRanker(String modelPath, String[] params) throws IOException{
    assert(params.length == 1);
    Reader reader = new FileReader(modelPath);
    return Ranker.RankerFactory.getFromModel(params[0], reader) ;
  }

  public static Config createOptionalConfig(String configPath, CommandLine line) throws IOException{
    ObjectMapper mapper = new ObjectMapper();
    Config optionalConfig = mapper.readValue(new File(configPath), SavedModel.class).config;

    if(line.hasOption("test"))
      optionalConfig.dataSet.test = line.getOptionValue("test");
    if(line.hasOption("report"))
      optionalConfig.report.file = line.getOptionValue("report");
    if(line.hasOption("eval"))
      optionalConfig.evaluation.evaluator = line.getOptionValue("eval");
    if(line.hasOption("k"))
      optionalConfig.evaluation.params.put("k", Integer.parseInt(line.getOptionValue("k")));

    return optionalConfig;
  }

  public static void evaluate(Ranker ranker, List<Query> testSet, Config optionalConfig){
    RankEval eval = RankEval.RankEvalFactory.get(optionalConfig.evaluation.evaluator);
    double score = eval.calculateAvgAllQueries(ranker, testSet, (int) optionalConfig.evaluation.params.get("k"));
    String header = optionalConfig.evaluation.evaluator + "@" + optionalConfig.evaluation.params.get("k") + " for " + optionalConfig.algorithm;
    Report report = Report.getReport((optionalConfig.report == null) ? null : optionalConfig.report.file, header);
    report.log(score);
    report.close();
  }

  private static class SavedModel { //TODO: Don't want to create another Saved Model...
    public Config config;
    public Object weights;    //These will not be used...
    public Object thresholds; //Will not be used...
    SavedModel(){  // this is needed for Jackson...
    }
    SavedModel(Config config){
      this.config = config;
    }
  }

}
