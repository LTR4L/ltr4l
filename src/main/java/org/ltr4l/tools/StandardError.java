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

public enum StandardError implements Error {
  SQUARE{
    @Override
    public double error(double output, double target) {
      return .5 * Math.pow(output - target, 2);
    }
    @Override
    public double der(double output, double target) {
      return output - target;
    }
  },

  ENTROPY{
    protected double DELTA = 1e-8;

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
  },

  FIDELITY{
    @Override
    public double error(double output, double target) {
      return 1 - (Math.sqrt(target * output) + Math.sqrt((1 - target) * (1 - output)));
    }
    @Override
    public double der(double output, double target) {
      return 1 / 2 * (Math.sqrt(target / output) + Math.sqrt((1 - target) / (1 - output)));
    }
  },
  HINGE{
    @Override
    public double error(double output, double target){
      return Math.max(1 - target * output, 0d); //Output should be W * X + b in linear case, target label
    }
    @Override
    public double der(double output, double target){ //Note: non-smoothed hinge loss
      return target * output < 1 ? -target : 0;
    }
  }
}
