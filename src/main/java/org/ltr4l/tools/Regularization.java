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

public interface Regularization {

  public static Type DEFAULT = Type.L2;

  double output(double weight);

  double derivative(double weight);

  public class L1 implements Regularization {

    @Override
    public double output(double weight) {
      return Math.abs(weight);
    }

    @Override
    public double derivative(double weight) {
      return weight > 0 ? 1 :
          weight < 0 ? -1 :
              0;
    }

  }

  public class L2 implements Regularization {

    @Override
    public double output(double weight) {
      return .5 * weight * weight;
    }

    @Override
    public double derivative(double weight) {
      return weight;
    }
  }

  public class RegularizationFactory {

    public static Regularization getRegularization(Type type) {
      switch (type) {
        case L1:
          return new L1();
        case L2:
          return new L2();
        default:
          return new L2();
      }
    }
  }

  public enum Type {
    L1, L2;
  }
}
