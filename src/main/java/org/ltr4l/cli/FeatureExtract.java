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

import org.apache.commons.cli.*;
import org.ltr4l.Version;
import org.ltr4l.click.*;
import org.ltr4l.lucene.solr.client.FeatureExtractor;

import java.io.*;
import java.lang.invoke.MethodHandles;

public class FeatureExtract {

  private static final String REQUIRED_ARG = "<solrUrl> <configFile> <impressionLogFile> <outputFile> <borders>";
  private static String outputFile;

  public static void main (String args[]) throws Exception{
    Options options = createOptions();
    CommandLine line = getCommandLine(options, args);

    if(line.hasOption("help")) printUsage(options);

    if(line.hasOption("version")){
      System.out.printf("%s release %s\n", MethodHandles.lookup().lookupClass().getCanonicalName(), Version.version);
      System.exit(0);
    }

    String[] params = line.getArgs();
    if(params == null || params.length == 0){
      System.err.printf("No required argument %s specified", REQUIRED_ARG);
      printUsage(options);
    }
    else if(params.length > 6){
      System.err.printf("Too many argument is specified: %s", params[6]);
      printUsage(options);
    }

    String url = line.hasOption("url") ? line.getOptionValue("url") : params[0];
    String configFile = line.hasOption("configFile") ? line.getOptionValue("configFile") : params[1];
    String impressionLogFile = line.hasOption("impressionLog") ? line.getOptionValue("impressionLog") : params[2];
    outputFile = line.hasOption("outputFile") ? line.getOptionValue("outputFile") : params[3];
    String bordersListStr = line.hasOption("borders") ? line.getOptionValue("borders") : params[4];
    String idField = null;

    if (params.length == 6) {
      idField = params[5];
    } else if (line.hasOption("idField")) {
      line.getOptionValue("idField");
    } else {
      idField = "id";
    }

    String trainingData = extract(url, configFile, impressionLogFile, bordersListStr, idField);
    writeTrainingData(trainingData);
  }

  private static Options createOptions(){
    Option help = new Option( "help", "print this message" );
    Option confFile = Option.builder("conf").argName("file").hasArg()
      .desc("use given file for configuration").build();
    Option verbose = new Option( "verbose", "be extra verbose" );
    Option noverbose = new Option( "noverbose", "override verboseness" );
    Option debug = new Option( "debug", "print debugging information" );

    Options options = new Options();
    options.addOption(help)
      .addOption(confFile)
      .addOption(verbose)
      .addOption(noverbose)
      .addOption(debug);
    return options;
  }

  private static void printUsage(Options options){
    HelpFormatter formatter = new HelpFormatter();
    formatter.printHelp( REQUIRED_ARG,
      "\nExecute feature extraction with Apache Solr. Required argument json file name for job configuration.\n\n",
      options, null, true );
    System.exit(0);
  }

  private static CommandLine getCommandLine(Options options, String[] args){
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

  private static String extract(String url, String confFile, String impressionLogFileName, String bordersListStr, String idField) throws Exception {
    File impressionLogFile = new File(impressionLogFileName);
    if (!impressionLogFile.canRead()) {
      System.err.printf("Cannot read impression log file: %s\n\n", impressionLogFile);
      System.exit(-1);
    }

    CMQueryHandler cmQueryHandler = new CMQueryHandler(new FileInputStream(impressionLogFile));

    FeatureExtractor featureExtractor = new FeatureExtractor(url, confFile, cmQueryHandler, idField);
    featureExtractor.execute();
    return featureExtractor.getMSFormatTrainingData(bordersListStr);
  }

  private static void writeTrainingData(String trainingData) {
    try (FileWriter fw = new FileWriter(new File(outputFile))) {
      fw.write(trainingData);
    } catch (IOException ioe) {
      ioe.printStackTrace();
      System.err.println("Exception happened while writing training data");
      System.exit(-1);
    }
  }
}
