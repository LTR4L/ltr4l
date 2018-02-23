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

import org.ltr4l.nn.Activation;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;

public class Config {

  private final int numIterations;
  private final double learningRate;
  private final Optimizer.OptimizerFactory optFact;
  private final Regularization reguFunction;
  private final String weightInit;
  private double reguRate;
  private final Object[][] networkShape;
  private final double bernNum;
  private final int PNum;
  private final String name;

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
    Properties props = new Properties();
    props.load(reader);

    name = getReqStrProp(props, "name");
    numIterations = getIntProp(props, "numIterations", 100);
    learningRate = getDoubleProp(props, "learningRate", 0);   // TODO: default value 0 is correct??
    optFact = chooseOptFact(props);
    reguFunction = Regularization.RegularizationFactory.getRegularization(getStrProp(props, "reguFunction", "L2"));
    weightInit = getStrProp(props, "weightInit", "zero");   // TODO: default value "zero" is correct??
    reguRate = getDoubleProp(props, "reguRate", 0); // TODO: default value 0 is correct??
    networkShape = parseLayers(props);
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

  private Optimizer.OptimizerFactory chooseOptFact(Properties props) {
    String opt = props.getProperty("optimizer");
    if(opt == null) return null;
    else{
      switch (opt.toLowerCase()) {
        case "adam":
          return new Optimizer.AdamFactory();
        case "sgd":
          return new Optimizer.sgdFactory();
        case "momentum":
          return new Optimizer.MomentumFactory();
        case "nesterov":
          return new Optimizer.NesterovFactory();
        case "adagrad":
          return new Optimizer.AdagradFactory();
        default:
          return null;
      }
    }
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
    if (reguFunction == null) {
      System.err.println("No regularization specified, default will be L2.");
      return Regularization.RegularizationFactory.getRegularization("L2");
    }
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
      return new Optimizer.sgdFactory();
    }
    return optFact;
  }

  private Object[][] parseLayers(Properties props){
    String value = props.getProperty("layers");
    if(value == null){
      return new Object[][]{{1, new Activation.Identity()}};
    }
    else{
      String[] layersInfo = value.split(" ");
      Object[][] nShape = new Object[layersInfo.length][2];
      for (int i = 0; i < layersInfo.length; i++) {
        String[] layerShape = layersInfo[i].split(",");
        Integer nodeNum = Integer.parseInt(layerShape[0]);
        //Default number of nodes is 1
        if (nodeNum == null || nodeNum < 0) {
          nodeNum = 1;
        }
        if (Activation.ActivationFactory.getActivator(layerShape[1]) == null)
          nShape[i] = new Object[]{nodeNum, new Activation.Identity()};

        else
          nShape[i] = new Object[]{Integer.parseInt(layerShape[0]), Activation.ActivationFactory.getActivator(layerShape[1])};
      }
      return nShape;
    }
  }

  public Object[][] getNetworkShape() {
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
}
