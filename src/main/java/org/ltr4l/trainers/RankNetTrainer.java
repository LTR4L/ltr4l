package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.RankNetMLP;
import org.ltr4l.nn.Regularization;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class RankNetTrainer extends MLPTrainer {
  protected RankNetMLP rmlp;
  protected List<Document[][]> trainingPairs;
  protected List<Document[][]> validationPairs;

  RankNetTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config, true);
    int featureLength = trainingSet.get(0).getFeatureLength();
    Object[][] networkShape = config.getNetworkShape();
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    rmlp = new RankNetMLP(featureLength, networkShape, optFact, regularization, weightModel);
    mlp = rmlp;

    trainingPairs = new ArrayList<>();
    for (int i = 0; i < trainingSet.size(); i++) {
      Query query = trainingSet.get(i);
      Document[][] documentPairs = query.orderDocPairs();
      trainingPairs.add(documentPairs);                   //add even if null, as placeholder for query.
    }

    validationPairs = new ArrayList<>();
    for (int i = 0; i < validationSet.size(); i++) {
      Query query = validationSet.get(i);
      Document[][] documentPairs = query.orderDocPairs();
      validationPairs.add(documentPairs);
    }
  }

  @Override
  public double calculateLoss(List<Query> queries) {
    List<Document[][]> docPairs;
    if (queries == trainingSet)
      docPairs = trainingPairs;
    else if (queries == validationSet)
      docPairs = validationPairs;
    else
      return -1d;
    double loss = 0d;
    for (Document[][] query : docPairs) {
      if (query == null)
        continue;
      //loss += Arrays.stream(query).mapToDouble( pair -> new ENTROPY().error(Math.pow(1 + Math.exp(rmlp.forwardProp(pair[1]) - rmlp.forwardProp(pair[0])), -1), 1d)).sum() / query.length;
      double queryLoss = 0d;
      for (Document[] pair : query) {
        double s1 = rmlp.forwardProp(pair[0]);
        double s2 = rmlp.forwardProp(pair[1]);
        double output = Math.pow(1 + Math.exp(s2 - s1), -1);
        queryLoss += new Error.ENTROPY().error(output, 1d) / (double) query.length;
      }
      loss += queryLoss;
    }
    return loss / (double) docPairs.size();
  }

  @Override
  public void train() {
    double threshold = 0.5;

    //Present all docs of randomly selected query
    //For number of queries / 6 times.
    for (int i = 0; i < trainingPairs.size() / 6; i++) {
      int iq = new Random().nextInt(trainingPairs.size());
      if (trainingPairs.get(iq) == null) {
        i--;
        continue;
      }
      for (Document[] docPair : trainingPairs.get(iq)) { //for each document pair in query iq
        Document docA = docPair[0];
        Document docB = docPair[1];

        double si = rmlp.forwardProp(docA);
        double sj = rmlp.forwardProp(docB);
        double delta = si - sj;

        if (delta < threshold) {
          double sigma = new Activation.Sigmoid().output(-delta);
          rmlp.backProp(sigma);
          rmlp.forwardProp(docA);
          rmlp.backProp(-sigma);
          rmlp.updateWeights(lrRate, rgRate);
        }
      }
    }
  }
}

