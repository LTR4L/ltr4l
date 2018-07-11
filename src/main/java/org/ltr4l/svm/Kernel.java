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
package org.ltr4l.svm;

import java.util.List;
import java.util.Objects;

/**
 * Kernel and its implementations for use in SVM algorithms.
 * U and V are features.
 */
public interface Kernel {

  public default double similarityK(List<Double> U, List<Double> V){
    return similarityK(U, V, new KernelParams());
  }

  public double similarityK(List<Double> U, List<Double> V, KernelParams params);

  public static Kernel getKernel(String kernelType){
    Objects.requireNonNull(kernelType);
    for(Kernel.Type type : Kernel.Type.values())
      if(type.name().equals(kernelType.toUpperCase()))
        return type;
    return Type.LINEAR;
  }

  public static enum Type implements Kernel {
    IDENTITY{
      @Override
      public double similarityK(List<Double> U, List<Double> V) {
        return VectorMath.dot(U, V);
      }
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params) {
        return similarityK(U, V);
      }

    },
    LINEAR{
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params) {
        return VectorMath.dot(U, V) + params.getC();
      }

    },
    POLYNOMIAL{
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params) {
        return Math.pow((params.getSigma() * VectorMath.dot(U, V)) + params.getC(), params.getD());
      }
    },
    GAUSSIAN{
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params)  {
        List<Double> diff = VectorMath.diff(U, V);
        return Math.exp(- (VectorMath.norm2(diff)) / (2 * params.getSigma() * params.getSigma()));
      }
    },
    EXPONENTIAL{
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params) {
        List<Double> diff = VectorMath.diff(U, V);
        return Math.exp(- (VectorMath.norm(diff)) / (2 * params.getSigma() * params.getSigma()));
      }
    },
    LAPLACIAN{
      @Override
      public double similarityK(List<Double> U, List<Double> V, KernelParams params) {
        List<Double> diff = VectorMath.diff(U, V);
        return Math.exp(- (VectorMath.norm(diff)) / (params.getSigma()));
      }
    }
  }

}
