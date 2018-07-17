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

import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.ltr4l.nn.AbstractMLP;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.MLP;
import org.ltr4l.nn.NetworkShape;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.WeightInitializer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.*;
import org.ltr4l.tools.Error;

/**
 * The basic implementation of AbstractTrainer for classes which use Multi-Layer Perceptron rankers.
 * As the training method can be different depending on the algorithm used,
 * the method train() must be implemented by child classes.
 */
public abstract class MLPTrainer<M extends AbstractMLP> extends AbstractTrainer<M, MLPTrainer.MLPConfig> {
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;
  //protected Config config;

  MLPTrainer(List<Query> training, List<Query> validation, MLPConfig config, M ranker) {
    this(training, validation, config, ranker, StandardError.SQUARE, new PointwiseLossCalc.StandardPointLossCalc<M>(training, validation, StandardError.SQUARE));
  }

  MLPTrainer(List<Query> training, List<Query> validation, MLPConfig config, M ranker, Error errorFunc, LossCalculator<M> lossCalc) {
    super(training, validation, config, ranker, errorFunc, lossCalc);
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
  }

  @Override
  protected AbstractMLP constructRanker(){
    int featureLength = trainingSet.get(0).getFeatureLength();
    NetworkShape networkShape = config.getNetworkShape();
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    return new MLP(featureLength, networkShape, optFact, regularization, weightModel);
  }

  @Override
  public Class<MLPConfig> getConfigClass(){
    return getCC();
  }

  static Class<MLPConfig> getCC(){
    return MLPConfig.class;
  }

  public static class MLPConfig extends Config {

    @JsonIgnore
    public double getLearningRate(){
      return getReqDouble(params, "learningRate");
    }

    @JsonIgnore
    public Map<String, Object> getRegularization(){
      return getReqParams(params, "regularization");
    }

    @JsonIgnore
    public double getReguRate(){
      return getReqDouble(getRegularization(), "rate");
    }

    @JsonIgnore
    public NetworkShape getNetworkShape(){
      List<Map<String, Object>> layers = getReqArrayParams(params, "layers");
      List<NetworkShape.LayerSetting> layerSettings = new ArrayList<>();
      for(Map<String, Object> params: layers){
        int num = getReqInt(params, "num");
        String activation = getString(params, "activator", "identity");
        Activation actFunc = Activation.ActivationFactory.getActivator(activation);
        layerSettings.add(new NetworkShape.LayerSetting(num, actFunc));
      }

      return new NetworkShape(layerSettings);
    }

    @JsonIgnore
    public Optimizer.OptimizerFactory getOptFact(){
      Optimizer.Type optType = Optimizer.Type.valueOf(getString(params, "optimizer", Optimizer.DEFAULT.name()));
      return Optimizer.getFactory(optType);
    }

    @JsonIgnore
    public Regularization getReguFunction(){
      Regularization.Type reguType = Regularization.Type.valueOf(getString(getRegularization(), "regularizer", Regularization.DEFAULT.name()));
      return Regularization.RegularizationFactory.getRegularization(reguType);
    }

    @JsonIgnore
    public String getWeightInit(){
      return getString(params, "weightInit", WeightInitializer.DEFAULT.name());
    }
  }
}
