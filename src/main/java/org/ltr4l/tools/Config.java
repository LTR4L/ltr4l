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

package org.ltr4l.tools;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.Properties;

import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.WeightInitializer;

public class Config {

  private final int numIterations;
  private final double learningRate;
  private final Optimizer.OptimizerFactory optFact;
  private final Regularization reguFunction;
  private final String weightInit;
  private final double reguRate;
  private final NetworkShape networkShape;
  private final double bernNum;
  private final int PNum;
  private final String name;
  private final Properties props;

  public static Config get(String file){
    try(InputStream is = new FileInputStream(file)){
      try(Reader reader = new InputStreamReader(is)){
        Config configs = get(reader);
        return configs;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static Config get(Reader reader) throws IOException {
    return new Config(reader);
  }

  private Config(Reader reader) throws IOException {
    props = new Properties(); //Changed to class variable for printing model.
    props.load(reader);

    name = getReqStrProp(props, "name");
    numIterations = getIntProp(props, "numIterations", 100);
    learningRate = getDoubleProp(props, "learningRate", 0);   // TODO: default value 0 is correct??
    Optimizer.Type optType = Optimizer.Type.valueOf(getStrProp(props, "optimizer", Optimizer.DEFAULT.name()));
    optFact = Optimizer.getFactory(optType);
    Regularization.Type reguType = Regularization.Type.valueOf(getStrProp(props, "reguFunction", Regularization.DEFAULT.name()));
    reguFunction = Regularization.RegularizationFactory.getRegularization(reguType);
    weightInit = getStrProp(props, "weightInit", WeightInitializer.DEFAULT.name());
    reguRate = getDoubleProp(props, "reguRate", 0); // TODO: default value 0 is correct??
    networkShape = NetworkShape.parseSetting(props.getProperty("layers"));
    bernNum = getDoubleProp(props, "bernoulli", 0.03);
    PNum = getIntProp(props, "N", 1);   // TODO: default value 1 is appropriate?
  }

  static String getStrProp(Properties props, String name, String defValue){
    String value = props.getProperty(name);
    return value == null ? defValue : value;
  }

  static String getReqStrProp(Properties props, String name){
    String value = props.getProperty(name);
    if(value == null) throw new IllegalArgumentException(String.format("parameter \"%s\" must be set in config file", name));
    return value;
  }

  static int getIntProp(Properties props, String name, int defValue){
    String value = props.getProperty(name);
    return value == null ? defValue : Integer.parseInt(value);
  }

  static int getReqIntProp(Properties props, String name){
    String value = props.getProperty(name);
    if(value == null) throw new IllegalArgumentException(String.format("parameter \"%s\" must be set in config file", name));
    return Integer.parseInt(value);
  }

  static double getDoubleProp(Properties props, String name, double defValue){
    String value = props.getProperty(name);
    return value == null ? defValue : Double.parseDouble(value);
  }

  static double getReqDoubleProp(Properties props, String name){
    String value = props.getProperty(name);
    if(value == null) throw new IllegalArgumentException(String.format("parameter \"%s\" must be set in config file", name));
    return Double.parseDouble(value);
  }

  public double getLearningRate() {
    return learningRate;
  }

  public double getReguRate() {
    return reguRate;
  }

  public int getNumIterations() {
    return numIterations;
  }

  public Regularization getReguFunction() {
    return reguFunction;
  }

  public String getWeightInit() {
    //Default Weight initialization to be determined in NN Constructor
    return weightInit;
  }

  //Have
  public Optimizer.OptimizerFactory getOptFact() {
    if (optFact == null) {
      System.err.println("No or invalid optimizer specified. Will use default SGD.");
      return new Optimizer.SGDFactory();
    }
    return optFact;
  }

  public NetworkShape getNetworkShape() {
    return networkShape;
  }

  public double getBernNum() {
    return bernNum;
  }

  public int getPNum() {
    return PNum;
  }

  public String getName() {
    return name;
  }

  public Properties getProps() {
    return props;
  }
}
