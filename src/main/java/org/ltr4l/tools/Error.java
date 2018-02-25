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

public interface Error {

  double error(double output, double target);

  double der(double output, double target);

  public class Square implements Error {

    @Override
    public double error(double output, double target) {
      return .5 * Math.pow(output - target, 2);
    }

    @Override
    public double der(double output, double target) {
      return output - target;
    }

  }

  /*
  public class Entropy implements Error {

    @Override
    public double error(double output, double target) {
      return -target * (Math.log(output)) - (1 - target) * Math.log(1 - output);
    }

    @Override
    public double der(double output, double target) {
      return (-target / output) + ((1 - target) / (1 - output));
    }
  }
  */

  /**
   * Calculate cross entropy error with <a href="https://en.wikipedia.org/wiki/One-hot">one-hot encoding</a>.
   * To avoid {@link java.lang.Double#NaN}, we always add a small value {@link #DELTA} to #output parameter.
   * TODO: but if output == -DELTA, the result could be NaN...
   */
  public class Entropy implements Error {

    static final double DELTA = 1e-8;

    @Override
    public double error(double output, double target) {
      assert(output >= 0);
      return -target * Math.log(output + DELTA);
    }

    @Override
    public double der(double output, double target) {
      assert(output >= 0);
      return -target / (output + DELTA);
    }
  }

  public class Fidelity implements Error {

    @Override
    public double error(double output, double target) {
      return 1 - (Math.sqrt(target * output) + Math.sqrt((1 - target) * (1 - output)));
    }

    @Override
    public double der(double output, double target) {
      return 1 / 2 * (Math.sqrt(target / output) + Math.sqrt((1 - target) / (1 - output)));
    }
  }
}

