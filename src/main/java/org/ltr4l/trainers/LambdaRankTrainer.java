package org.ltr4l.trainers;

import org.ltr4l.nn.Activation;
import org.ltr4l.tools.Config;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;

public class LambdaRankTrainer extends RankNetTrainer {

  LambdaRankTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config);
  }

  @Override
  public void train() {
    for (int iq = 0; iq < trainingSet.size(); iq++) {
      if (trainingPairs.get(iq) == null)
        continue;

      Query query = trainingSet.get(iq);

      HashMap<Document, Double> ranks = new HashMap<>();
      HashMap<Document, Double> lambdas = new HashMap<>();
      HashMap<Document, Double> pws = new HashMap<>();
      HashMap<Document, Double> logs = new HashMap<>();
      double N = idcg(query.getDocList(), query.getDocList().size());


      List<Document> sorted = sortP(query);

      for (int i = 0; i < sorted.size(); i++) {
        Document doc = sorted.get(i);
        ranks.put(doc, rmlp.forwardProp(doc));
        lambdas.put(doc, 0d);
        pws.put(doc, Math.pow(2, doc.getLabel()) - 1);
        logs.put(doc, 1 / Math.log(i + 2));
      }

      for (Document[] pair : trainingPairs.get(iq)) {
        double dNCG = N * (pws.get(pair[0]) - pws.get(pair[1])) * (logs.get(pair[0]) - logs.get(pair[1]));
        double diff = ranks.get(pair[1]) - ranks.get(pair[0]);  //- (si - sj)
        double lambda = Math.abs(new Activation.Sigmoid().output(diff) * dNCG);
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

  private double idcg(List<Document> docList, int position) {
    List<Document> docsRanks = new ArrayList<>(docList);
    docsRanks.sort(Comparator.comparingInt(Document::getLabel).reversed());
    double sum = 0;
    if (position > -1) {
      final int pos = Math.min(position, docsRanks.size());
      for (int i = 0; i < pos; i++) {
        sum += (Math.pow(2, docsRanks.get(i).getLabel()) - 1) / Math.log(i + 2);
      }
    }
    return sum * Math.log(2);  //Change of base
  }


}