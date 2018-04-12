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


import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * This class is responsible for parsing the file for the required parameters for training and holding the parameters.
 */
public class Config {

  public String algorithm;
  public int numIterations = 100;
  public int batchSize;
  public boolean verbose;
  public boolean nomodel;

  public Map<String, Object> params;
  public Config.DataSet dataSet;
  public Config.Model model;
  public Config.Evaluation evaluation;
  public Config.Report report;

  public Config overrideBy(Config override){
    if(override != null){
      this.algorithm = override.algorithm;
      this.numIterations = override.numIterations;
      this.batchSize = override.batchSize;
      this.verbose = override.verbose;
      this.nomodel = override.nomodel;
      this.params = override.params;
      this.dataSet = override.dataSet;
      this.model = override.model;
      this.evaluation = override.evaluation;
      this.report = override.report;
    }

    return this;
  }

  public static class DataSet {
    public String training;
    public String validation;
    public String test;
  }

  public static class Model {
    public static final String DEFAULT_MODEL_FILE = "model/model.txt";
    public String format;
    public String file;
  }

  public static class Evaluation {
    public String evaluator;
    public Map<String, Object> params;
  }

  public static class Report {
    public String format;
    public String file;
  }

  public static String getReqString(Map<String, Object> params, String name){
    Object obj = params.get(name);
    return Objects.requireNonNull(obj, name + " must be set in params!").toString();
  }

  public static String getString(Map<String, Object> params, String name, String defValue){
    Object obj = params.get(name);
    if(obj == null){
      return defValue;
    }
    return obj.toString();
  }

  public static int getReqInt(Map<String, Object> params, String name){
    Object obj = params.get(name);
    return Integer.parseInt(Objects.requireNonNull(obj, name + " must be set in params!").toString());
  }

  public static int getInt(Map<String, Object> params, String name, int defValue){
    Object obj = params.get(name);
    if(obj == null){
      return defValue;
    }
    return Integer.parseInt(obj.toString());
  }

  public static double getReqDouble(Map<String, Object> params, String name){
    Object obj = params.get(name);
    return Double.parseDouble(Objects.requireNonNull(obj).toString());
  }

  public static double getDouble(Map<String, Object> params, String name, double defValue){
    Object obj = params.get(name);
    if(obj == null){
      return defValue;
    }
    return Double.parseDouble(obj.toString());
  }

  public static Map<String, Object> getReqParams(Map<String, Object> params, String name){
    Object obj = params.get(name);
    return (Map<String, Object>)Objects.requireNonNull(obj, name + " must be set in params!");
  }

  public static List<Map<String, Object>> getReqArrayParams(Map<String, Object> params, String name){
    Object obj = params.get(name);
    return (List<Map<String, Object>>)Objects.requireNonNull(obj, name + " must be set in params!");
  }
}
