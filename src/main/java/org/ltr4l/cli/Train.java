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

import java.io.File;
import java.io.IOException;
import java.lang.invoke.MethodHandles;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.DefaultParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.Option;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.ltr4l.Version;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.trainers.AbstractTrainer;

public class Train {

  private static final String REQUIRED_ARG = "<LTR-algorithm-name>";

  public static void main(String[] args) throws Exception {

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

    String configPath = getConfigPath(line, params);
    Config optionalConfig = createOptionalConfig(configPath, line);

    QuerySet trainingSet = QuerySet.create(optionalConfig.dataSet.training);
    QuerySet validationSet = QuerySet.create(optionalConfig.dataSet.validation);

    AbstractTrainer trainer = AbstractTrainer.TrainerFactory.getTrainer(params[0], trainingSet, validationSet, configPath, optionalConfig);
    long startTime = System.currentTimeMillis();
    trainer.trainAndValidate();
    long endTime = System.currentTimeMillis();
    System.out.println("Took " + (endTime - startTime) + " ms to complete epochs.");
  }

  public static Options createOptions(){
    Option help = new Option( "help", "print this message" );
    Option configFile = Option.builder("config").argName("file").hasArg()
        .desc("use given file for configuration").build();
    Option numIte = Option.builder("iterations").argName("num").hasArg()
        .desc("use given number of iterations").build();
    Option trainDataSet = Option.builder("training").argName("file").hasArg()
        .desc("use given file for training").build();
    Option validDataSet = Option.builder("validation").argName("file").hasArg()
        .desc("use given file for validation").build();
    Option modelFile = Option.builder("model").argName("file").hasArg()
        .desc("specify model file name").build();
    Option reportFile = Option.builder("report").argName("file").hasArg()
        .desc("specify report file name").build();
    Option version = new Option( "version", "print the version information and exit" );
    Option verbose = new Option( "verbose", "be extra verbose" );
    Option noverbose = new Option( "noverbose", "override verboseness" );
    Option debug = new Option( "debug", "print debugging information" );

    Options options = new Options();
    options.addOption(help)
        .addOption(configFile)
        .addOption(numIte)
        .addOption(trainDataSet)
        .addOption(validDataSet)
        .addOption(modelFile)
        .addOption(reportFile)
        .addOption(version)
        .addOption(verbose)
        .addOption(noverbose)
        .addOption(debug);
    return options;
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

  public static void printUsage(Options options){
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( "train " + REQUIRED_ARG,
        "\nExecute Learning-to-Rank training algorithm. The algorithm is specified by the required argument <LTR-algorithm-name>." +
            " The program will look for the configuration file \"config/<LTR-algorithm-name>.json\"" +
            " unless config option is specified." +
            " The following options can be specified in order to override the existing settings in the config file.\n\n",
        options, null, true );
    System.exit(0);
  }

  public static String getConfigPath(CommandLine line, String[] params){
    assert(params.length == 1);
    return line.hasOption("config") ? line.getOptionValue("config") : String.format("confs/%s.json", params[0]);
  }

  public static Config createOptionalConfig(String configPath, CommandLine line) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    Config optionalConfig = mapper.readValue(new File(configPath), Config.class);

    if(line.hasOption("iterations"))
      optionalConfig.numIterations = Integer.parseInt(line.getOptionValue("iterations"));
    if(line.hasOption("verbose"))
      optionalConfig.verbose = true;
    if(line.hasOption("noverbose"))
      optionalConfig.verbose = false;    // noverbose overrides verboseness
    if(line.hasOption("training"))
      optionalConfig.dataSet.training = line.getOptionValue("training");
    if(line.hasOption("validation"))
      optionalConfig.dataSet.validation = line.getOptionValue("validation");
    if(line.hasOption("model"))
      optionalConfig.model.file = line.getOptionValue("model");
    if(line.hasOption("report"))
      optionalConfig.report.file = line.getOptionValue("report");

    return optionalConfig;
  }
}
