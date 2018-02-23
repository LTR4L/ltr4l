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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;

import org.ltr4l.nn.Activation;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;

public class Config {
  private int numIterations;
  private double learningRate;
  private Optimizer.OptimizerFactory optFact;
  private Regularization reguFunction;
  private String weightInit;
  private double reguRate;
  private Object[][] networkShape;
  private double bernNum;
  private int PNum;
  private String name;

  public static Config get(String file){
    try(InputStream is = new FileInputStream(file)){
      try(Reader reader = new InputStreamReader(is)){
        Config configs = new Config(reader);
        return configs;
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private Config(Reader reader) throws IOException {
    numIterations = -1;
    learningRate = -1;
    optFact = null;
    reguFunction = null;
    weightInit = null;
    reguRate = -1;
    networkShape = null;
    bernNum = 0d;
    PNum = -1;


    BufferedReader br = new BufferedReader(reader);
    String line = "";

    while (line != null) {
      line = br.readLine();
      if (line != null && !line.equals("")) {
        String[] parameter = line.split(":");
        if (parameter.length == 1 || parameter.length < 2)
          continue;
        switch (parameter[0]) {
          case "name":
            name = parameter[1];
            break;
          case "numIterations":
            numIterations = Integer.parseInt(parameter[1]);
            break;
          case "learningRate":
            learningRate = Double.parseDouble(parameter[1]);
            break;
          case "optimizer":
            optFact = chooseOptFact(parameter[1]);
            break;
          case "weightInit":
            weightInit = parameter[1];
            break;
          case "reguFunction":
            reguFunction = Regularization.RegularizationFactory.getRegularization(parameter[1]);
            break;
          case "reguRate":
            reguRate = Double.parseDouble(parameter[1]);
            break;
          case "layers":
            String[] layersInfo = parameter[1].split(" ");
            networkShape = new Object[layersInfo.length][2];
            for (int i = 0; i < layersInfo.length; i++) {
              String[] layerShape = layersInfo[i].split(",");
              Integer nodeNum = Integer.parseInt(layerShape[0]);
              //Default number of nodes is 1
              if (nodeNum == null || nodeNum < 0) {
                nodeNum = 1;
              }
              if (Activation.ActivationFactory.getActivator(layerShape[1]) == null)
                networkShape[i] = new Object[]{nodeNum, new Activation.Identity()};

              else
                networkShape[i] = new Object[]{Integer.parseInt(layerShape[0]), Activation.ActivationFactory.getActivator(layerShape[1])};
            }
            break;
          case "bernoulli":
            bernNum = Double.parseDouble(parameter[1]);
            break;
          case "N":
            PNum = Integer.parseInt(parameter[1]);
            break;
          default:
            break;
        }
      }
    }
  }

  private Optimizer.OptimizerFactory chooseOptFact(String opt) {
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

  public double getLearningRate() {
    return learningRate < 0 ? 0 : learningRate;
  }

  public double getReguRate() {
    return reguRate < 0 ? 0 : reguRate;
  }

  public int getNumIterations() {
    return numIterations <= 0 ? 100 : numIterations;
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

  public Object[][] getNetworkShape() {
    if (networkShape == null) {
      return new Object[][]{{1, new Activation.Identity()}};
    }
    return networkShape;
  }

  public double getBernNum() {
    return bernNum <= 0d ? 0.03 : bernNum;
  }

  public int getPNum() {
    return PNum <= 0 ? 1 : PNum;
  }

  public String getName() {
    return name;
  }
}
