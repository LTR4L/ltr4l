package org.ltr4l.trainers;

import org.ltr4l.nn.Activation;
import org.ltr4l.tools.Config;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.HashMap;

public class FRankTrainer extends RankNetTrainer {


  FRankTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config);
  }

  @Override
  public void train() {
    for (int iq = 0; iq < trainingSet.size(); iq++) {  //index query
      if (trainingPairs.get(iq) == null)
        continue; //if there are no valid pairs for the query, skip.

      Query query = trainingSet.get(iq);
      //int qsize = query.getDocList().size();
      //double[] lambdas = new double[qsize];
      //double[] ranks = new double[qsize];
      HashMap<Document, Double> lambdas = new HashMap<>(); //lambdas
      HashMap<Document, Double> ranks = new HashMap<>();   //Create map for documents ranks.
      for (Document doc : query.getDocList()) {
        lambdas.put(doc, 0d);
        ranks.put(doc, rmlp.forwardProp(doc));
      }

      for (Document[] pair : trainingPairs.get(iq)) {
        double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj)
        double lambda = new Activation.Sigmoid().output(diff);
        lambdas.put(pair[0], lambdas.get(pair[0]) - lambda); //λ1 = λ1 - dλ
        lambdas.put(pair[1], lambdas.get(pair[1]) + lambda); //λ2 = λ2 - dλ
      }

      for (Document doc : query.getDocList()) {
        rmlp.forwardProp(doc);
        rmlp.backProp(lambdas.get(doc));
      }
    }
    rmlp.updateWeights(lrRate, rgRate);
  }

}