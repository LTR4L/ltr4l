package org.ltr4l.nn;

public interface Regularization {

  double output(double weight);

  double derivative(double weight);

  class L1 implements Regularization {

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

  class L2 implements Regularization {

    @Override
    public double output(double weight) {
      return .5 * weight * weight;
    }

    @Override
    public double derivative(double weight) {
      return weight;
    }
  }

  class RegularizationFactory {

    public static Regularization getRegularization(String regularization) {
      switch (regularization) {
        case "L1":
          return new L1();
        case "L2":
          return new L2();
        default:
          return null;
      }
    }
  }
}

