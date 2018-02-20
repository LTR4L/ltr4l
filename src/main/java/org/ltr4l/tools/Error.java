package org.ltr4l.tools;

public interface Error {

    double error(double output, double target);

    double der(double output, double target);


    class SQUARE implements Error {

        @Override
        public double error(double output, double target){
            return .5*Math.pow(output - target, 2);
        }

        @Override
        public double der(double output, double target){
            return output - target;
        }

    }

    class ENTROPY implements Error{

        @Override
        public double error(double output, double target) {
            if (output == 0 || output == 1)
                System.out.println("error " + output);
            return -target * (Math.log(output)) - (1 - target) * Math.log(1 - output);
        }

        @Override
        public double der(double output, double target) {
            return (-target / output) + ((1 - target)/(1 - output));
        }
    }

    class LISTENTROPY implements Error {

        @Override
        public double error(double output, double target) {
            return - target * Math.log(output);
        }

        @Override
        public double der(double output, double target) {
            return - target / output;
        }
    }

    class FIDELITY implements Error{

        @Override
        public double error(double output, double target) {
            return 1 - (Math.sqrt(target * output) + Math.sqrt((1 - target) * (1 - output)));
        }

        @Override
        public double der(double output, double target) {
            return 1/2 * (Math.sqrt(target / output) + Math.sqrt((1 - target) / (1 - output)));
        }
    }
}

