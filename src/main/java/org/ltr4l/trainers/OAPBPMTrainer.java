package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.*;

public class OAPBPMTrainer extends LTRTrainer {
  final private OAPBPMRank ranker;
  private double maxScore;

  OAPBPMTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config.getNumIterations());
    maxScore = 0d;
    ranker = new OAPBPMRank(trainingSet.get(0).getFeatureLength(), QuerySet.findMaxLabel(trainingSet), config.getPNum(), config.getBernNum());

  }

  @Override
  public void train() {
    //List<Query> tSet = new ArrayList<>(trainingSet);
    //Collections.shuffle(tSet);
    //Collections.shuffle(trainingSet);
    for (Query query : trainingSet) {
      //Collections.shuffle(query.getDocList());
      for (Document doc : query.getDocList()) {
        ranker.updateWeights(doc);
      }
    }
  }

  protected double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> new Error.SQUARE().error(ranker.predict(doc), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  @Override
  public List<Document> sortP(Query query) {
    //List<Document> ranks = new ArrayList<>(query.getDocList()); //PROBLEM?
    List<Document> ranks = query.getDocList();
    //Collections.shuffle(ranks);
    ranks.sort(Comparator.comparingInt(ranker::predict).reversed());  //to put in order of highest to lowest
    //ranks.sort((doc1, doc2) -> Integer.compare(predict(doc2), predict(doc1)));
    return ranks;
  }
}

class OAPBPMRank extends PRank {
  private List<PRank> pRanks;
  private final double bernProb;

  OAPBPMRank(int featureLength, int maxLabel, int pNumber, double bernNumber) {
    super(featureLength, maxLabel);
    pRanks = new ArrayList<>();
    for (int i = 0; i < pNumber; i++)
      pRanks.add(new PRank(featureLength, maxLabel));
    bernProb = bernNumber; //Note: must be between 0 and 1.
  }

  @Override
  public void updateWeights(Document doc) {
    for (PRank prank : pRanks) {
      //Will or will not present document to the perceptron.
      if (bernoulli() == 1) {
        int prediction = prank.predict(doc);
        int label = doc.getLabel();
        if (label != prediction) { //if the prediction is wrong, update that perceptron's weights
          prank.updateWeights(doc);
          for (int i = 0; i < weights.length; i++)
            weights[i] += prank.getWeights()[i] / (double) pRanks.size(); //and update overall weights
          for (int i = 0; i < thresholds.length; i++)
            thresholds[i] += prank.getThresholds()[i] / (double) pRanks.size();
        }
      }

    }
  }

  private int bernoulli() {
    return new Random().nextDouble() < bernProb ? 1 : 0;
  }

}
