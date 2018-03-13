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

/**
 * The variants of gradient descent which optimize the parameter update.
 * Note that the standard gradient descent is of the form w = w - η(dw).
 */
public interface Optimizer {

  public static final Type DEFAULT = Type.sgd;

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
      r = beta2 * r + (1 - beta2) * dw * dw;
      double mH = m / (1 - Math.pow(beta1, iter));
      double rH = r / (1 - Math.pow(beta2, iter)); //Note: should always be positive for 0 <= beta2 < 1

      return -rate * mH / (Math.sqrt(rH) + eps);
    }
  }

  class AdamFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Adam();
    }
  }

  class SGD implements Optimizer {

    @Override
    public double optimize(double dw, double rate, long iter) {
      return -rate * dw;
    }
  }

  class SGDFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new SGD();
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

  class MomentumFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Momentum();
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
      cache = cache + (dw * dw);
      return - rate * dw / (Math.sqrt(cache) + eps);
    }
  }

  class AdagradFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Adagrad();
    }
  }

  class RMSProp implements Optimizer{
    private double cache;
    private static final double eps = 1e-6;   //Note: different values possible
    private static final double decay = 0.99; //Note: different values possible

    RMSProp(){
      cache = 0;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      cache = decay * cache + (1 - decay) * dw * dw;
      return - (rate * dw)/Math.sqrt(cache + eps);
    }
  }

  class RMSFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new RMSProp();
    }
  }

  class Adamax implements Optimizer { //Recommended rate is 0.002
    private final double beta1;
    private final double beta2;
    private double m;
    private double r; //r used instead of v for consistency with implementation of Adam.

    Adamax(){
      beta1 = 0.9;
      beta2 = 0.999;
      m = 0.0;
      r = 0.0;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      m = (beta1 * m) + (1 - beta1) * dw;
      r = Math.max(beta2 * r, Math.abs(dw));
      return rate * m / r;
    }
  }

  class AdamaxFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Adamax();
    }
  }

  class Nadam implements Optimizer {
    private final double beta1;
    private final double beta2;
    private double m;
    private double r;
    private double eps;

    Nadam(){
      beta1 = 0.9;
      beta2 = 0.999;
      m = 0.0;
      r = 0.0;
      eps = 1e-8;
    }

    @Override
    public double optimize(double dw, double rate, long iter) {
      m = (beta1 * m) + (1 - beta1) * dw;
      r = (beta2 * r) + (1 - beta2) * dw * dw;
      double mH = m / (1 - Math.pow(beta1, iter));
      double rH = r / (1 - Math.pow(beta2, iter));

      return - rate * (beta1 * mH + ((1 - beta1) * dw / (1 - Math.pow(beta1, iter)))) / (Math.sqrt(rH) + eps);
    }
  }

  class NadamFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new Nadam();
    }
  }

  class AMSGrad implements Optimizer {
    private final double beta1;
    private final double beta2;
    private final double eps;
    private double m;
    private double r;
    private double rH;

    AMSGrad(){
      beta1 = 0.9;
      beta2 = 0.99;
      eps = 1e-8;
      m = 0.0;
      r = 0.0;
      rH = 0.0;
    }


    @Override
    public double optimize(double dw, double rate, long iter) {
      m = beta1 * m + (1 - beta1) * dw;
      r = beta2 * r + (1 - beta2) * dw * dw;
      rH = Math.max(rH, r);

      return - rate * m / (Math.sqrt(rH) + eps);
    }
  }

  class AMSGradFactory implements OptimizerFactory {

    @Override
    public Optimizer getOptimizer() {
      return new AMSGrad();
    }
  }

  public static Optimizer.OptimizerFactory getFactory(Type type) {
    switch (type) {
      case adam:
        return new Optimizer.AdamFactory();
      case sgd:
        return new Optimizer.SGDFactory();
      case momentum:
        return new Optimizer.MomentumFactory();
      case nesterov:
        return new Optimizer.NesterovFactory();
      case adagrad:
        return new Optimizer.AdagradFactory();
      case rmsprop:
        return new Optimizer.RMSFactory();
      case adamax:
        return new Optimizer.AdamaxFactory();
      case nadam:
        return new Optimizer.NadamFactory();
      case amsgrad:
        return new Optimizer.AMSGradFactory();
      default:
        return new Optimizer.SGDFactory();
    }
  }

  public enum Type {
    adam, sgd, momentum, nesterov, adagrad, rmsprop, adamax, nadam, amsgrad;
  }
}
