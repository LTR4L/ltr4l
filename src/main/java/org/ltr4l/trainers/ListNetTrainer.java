package org.ltr4l.trainers;

import org.ltr4l.nn.Activation;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.nn.ListNetMLP;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.nn.Regularization;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

public class ListNetTrainer extends LTRTrainer {
  private double lrRate;
  private double rgRate;
  private ListNetMLP lmlp;

  ListNetTrainer(QuerySet training, QuerySet validation, Config config) {
    this(training, validation, config, false);
  }

  //This constructor exists solely for the purpose of child classes
  //It gives child classes the ability to assign an extended MLP.
  ListNetTrainer(QuerySet training, QuerySet validation, Config config, boolean hasOtherMLP) {
    super(training, validation, config.getNumIterations());
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
    if (!hasOtherMLP) {
      int featureLength = trainingSet.get(0).getFeatureLength();
      Object[][] networkShape = Arrays.copyOf(config.getNetworkShape(), config.getNetworkShape().length + 1);
      networkShape[networkShape.length - 1] = new Object[]{1, new Activation.Identity()};
      Optimizer.OptimizerFactory optFact = config.getOptFact();
      Regularization regularization = config.getReguFunction();
      String weightModel = config.getWeightInit();
      lmlp = new ListNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
    }
  }

  @Override
  public void train() {
    for (Query query : trainingSet) {
      for (Document doc : query.getDocList()) {
        lmlp.forwardProp(doc);
        lmlp.backProp(doc.getLabel());
      }
      lmlp.updateWeights(lrRate, rgRate);
    }
  }

  @Override
  protected double calculateLoss(List<Query> querySet) {
    double loss = 0;
    for (Query query : querySet) {
      double targetSum = query.getDocList().stream().mapToDouble(i -> Math.exp(i.getLabel())).sum();
      double outputSum = query.getDocList().stream().mapToDouble(i -> Math.exp(lmlp.forwardProp(i))).sum();
      double qLoss = query.getDocList().stream().mapToDouble(i -> new Error.LISTENTROPY().error( //-Py(log(Pfx))
          Math.exp(lmlp.forwardProp(i)) / outputSum, //output: exp(f(x)) / sum(f(x))
          i.getLabel() / targetSum))                 //target: y / sum(exp(y))
          .sum(); //sum over all documents                // Should it be exp(y)/sum(exp(y))?
      loss += qLoss;
    }
    return loss / querySet.size();
  }

  @Override
  public List<Document> sortP(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    ranks.sort(Comparator.comparingDouble(lmlp::predict).reversed());
    return ranks;
  }
}

