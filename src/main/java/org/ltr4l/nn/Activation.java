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

public interface Activation {
  double output(double input);

  double derivative(double input);

  class Identity implements Activation {

    @Override
    public double output(double input) {
      return input;
    }

    @Override
    public double derivative(double input) {
      return 1;
    }
  }

  class Sigmoid implements Activation {

    @Override
    public double output(double input) {
      return 1 / (1 + Math.exp(-input));
    }

    @Override
    public double derivative(double input) {
      double output = output(input);
      return (output) * (1 - output);
    }
  }

  class ReLu implements Activation {

    @Override
    public double output(double input) {
      return Math.max(0.01, input);
    }

    @Override
    public double derivative(double input) {
      return input <= 0 ? 0 : 1;
    }
  }

  class ActivationFactory {
    public static Activation getActivator(String actName) {
      switch (actName.toLowerCase()) {
        case "identity":
          return new Identity();
        case "sigmoid":
          return new Sigmoid();
        case "relu":
          return new ReLu();
        default:
          return null;
      }
    }
  }
}

