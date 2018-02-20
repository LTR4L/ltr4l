package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.MLP;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.Arrays;

public class NNRankTrainer extends MLPTrainer {
  private final int outputNodeNumber;

  //Last layer of the network has a number of nodes equal to the number of categories.
  //That layer is created in the constructor, so it is not necessary to specify last layer in config file.
  NNRankTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config, true);
    outputNodeNumber = QuerySet.findMaxLabel(trainingSet) + 1;
    int featureLength = trainingSet.get(0).getFeatureLength();
    //Add an output layer with number of nodes equal to number of classes/relevance categories.
    Object[][] networkShape = Arrays.copyOf(config.getNetworkShape(), config.getNetworkShape().length + 1);
    networkShape[networkShape.length - 1] = new Object[]{outputNodeNumber, new Activation.Sigmoid()};
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();

    //As the structure of the output layer has changed, predict needs to be overridden.
    mlp = new MLP(featureLength, networkShape, optFact, regularization, weightModel) {
      @Override
      public double predict(Document doc) {
        double threshold = 0.5;
        forwardProp(doc);
        for (int nodeId = 0; nodeId < network.get(network.size() - 1).size(); nodeId++) {
          Node node = network.get(network.size() - 1).get(nodeId);
          if (node.getOutput() < threshold)
            return nodeId - 1;
        }
        return network.get(network.size() - 1).size() - 1;
      }
    };
  }

  private double[] targetLabel(int label) {
    double[] targets = new double[outputNodeNumber]; //initialized with 0.
    for (int index = 0; index <= label; index++)
      targets[index] = 1;
    return targets;
  }

  @Override
  public void train() {
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        int output = (int) mlp.predict(doc);
        int label = doc.getLabel();
        if (output != label) {
          mlp.backProp(targetLabel(label), new Error.SQUARE());
          mlp.updateWeights(lrRate, rgRate);
        }
      }
    }
  }
}
