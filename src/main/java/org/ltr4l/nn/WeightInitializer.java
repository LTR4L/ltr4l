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

package org.ltr4l.nn;

import java.util.Random;

public class WeightInitializer {

  public static final Type DEFAULT = Type.normal;
  private final Type type;
  private final int num;
  private final Random r;

  static WeightInitializer get(String type, int num){
    return new WeightInitializer(Type.valueOf(type), num);
  }

  public static WeightInitializer get(String weightModel, int inputDim, NetworkShape networkShape){
    return get(weightModel, getNumWeights(inputDim, networkShape));
  }

  public static int getNumWeights(int inputDim, NetworkShape networkShape){
    int nWeights = inputDim * networkShape.getLayerSetting(0).getNum();  //Number of weights used for Xavier initialization.
    for (int i = 1; i < networkShape.size(); i++) {
      nWeights += networkShape.getLayerSetting(i - 1).getNum() * networkShape.getLayerSetting(i).getNum();
    }
    return nWeights;
  }

  private WeightInitializer(Type type, int num){
    this.type = type;
    this.num = num;
    this.r = new Random(System.currentTimeMillis());
  }

  public double getNextRandomInitialWeight(){
    switch (type){
      case xavier: return r.nextGaussian() / num;
      case normal: return r.nextGaussian();
      case uniform: return r.nextDouble();
      case zero: return 0;
      default: return r.nextGaussian();
    }
  }

  public double getInitialBias(){
    if(type == Type.zero) return 0;
    else return 0.01;
  }

  public enum Type {
    xavier, normal, uniform, zero;
  }
}
