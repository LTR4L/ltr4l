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

public interface Optimizer {

  double optimize(double dw, double rate, long iter);

  interface OptimizerFactory {

    Optimizer getOptimizer();
  }

  class Adam implements Optimizer {
    private final double beta1;
    private final double beta2;
    private final double eps;
    private double m;
    private double r;

    Adam() {
      beta1 = 0.9;
      beta2 = 0.999;
      eps = 1e-8; //.00000008
      m = 0.0;
      r = 0.0;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      m = beta1 * m + (1 - beta1) * dw;
      r = beta2 * m + (1 - beta2) * dw * dw;

      return -rate * m / (Math.sqrt(Math.abs(r)) + eps);
    }
  }

  class AdamFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Adam();
    }
  }

  class sgd implements Optimizer {

    @Override
    public double optimize(double dw, double rate, long iter) {
      return -rate * dw;
    }
  }

  class sgdFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new sgd();
    }
  }

  class Momentum implements Optimizer {
    private final double beta;
    private double m;

    Momentum() {
      beta = 0.95;
      m = 0;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      m = beta * m - rate * dw;
      return m;
    }
  }

  class Nesterov implements Optimizer {
    private final double beta;
    private double mp;
    private double m;

    Nesterov() {
      beta = 0.95;
      mp = 0.00;
      m = 0.0;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      mp = m;
      m = beta * m - rate * dw;
      return -beta * mp + (1 + beta) * m;
    }
  }

  class MomentumFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Momentum();
    }
  }

  class NesterovFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Nesterov();
    }
  }

  class Adagrad implements Optimizer {
    private final double eps;
    private double cache;

    Adagrad(){
      cache = 0.0;
      eps = 1e-6;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      cache = dw * dw;
      return - rate * dw / (Math.sqrt(cache) + eps);
    }
  }

  class AdagradFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Adagrad();
    }
  }

}

