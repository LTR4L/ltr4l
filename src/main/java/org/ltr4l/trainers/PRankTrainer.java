package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.RankEval;
import org.ltr4l.tools.Report;

import java.util.*;
import java.util.stream.IntStream;

public class PRankTrainer extends LTRTrainer {
  final private PRank ranker;
  private double maxScore;

  PRankTrainer(QuerySet training, QuerySet validation, Config configs) {
    super(training, validation, configs.getNumIterations());
    maxScore = 0.0;
    ranker = new PRank(training.getFeatureLength(), QuerySet.findMaxLabel(trainingSet));
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

  public void validate(int iter, int pos) {
/*        Collections.shuffle(validationSet);
        for(Query query : validationSet){
            Collections.shuffle(query.getDocList());
        }*/
    double newScore = RankEval.ndcgAvg(this, validationSet, pos);
    if (newScore > maxScore) {
      maxScore = newScore;
    }
    double[] losses = calculateLoss();
    System.out.println(iter + "  " + newScore);
    Report.report(iter, newScore, losses[0], losses[1]);
  }

  protected double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      List<Document> docList = query.getDocList();
      loss += docList.stream().mapToDouble(doc -> new Error.SQUARE().error(ranker.predict(doc), doc.getLabel())).sum() / docList.size();
    }
    return loss / queries.size();
  }

  //Sort documents in a query based on current model.
  public List<Document> sortP(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    ranks.sort(Comparator.comparingInt(ranker::predict).reversed());  //to put in order of highest to lowest
    return ranks;
  }
}

class PRank {
  double[] weights;
  double[] thresholds;

  PRank(int featureLength, int maxLabel) {
    if (featureLength > 0 && maxLabel > 0) {
      weights = new double[featureLength];
      thresholds = new double[maxLabel];
    } else {
      weights = null;
      thresholds = null;
    }

  }

  public double[] getWeights() {
    return weights;
  }

  public double[] getThresholds() {
    return thresholds;
  }

  public void updateWeights(Document doc) {
    double wx = predictRelScore(doc);
    int output = predict(doc);
    int label = doc.getLabel();
    if (output == label)//if output == label, do not update weights.
      return;
    int[] tau = new int[thresholds.length];
    for (int r = 0; r <= thresholds.length - 1; r++) { /////thresholds.length ??
      int ytr;
      if (label <= r)
        ytr = -1;
      else
        ytr = 1;
      if ((wx - thresholds[r]) * ytr <= 0)
        tau[r] = ytr;
      else
        tau[r] = 0;
    }
    int T = IntStream.of(tau).sum();
    for (int i = 0; i <= weights.length - 1; i++) {
      weights[i] += T * doc.getFeatures().get(i);
    }
    for (int r = 0; r <= thresholds.length - 1; r++) {
      thresholds[r] -= tau[r];
    }
  }

  protected int predict(Document doc) {
    double wx = predictRelScore(doc);
    for (int i = 0; i < thresholds.length; i++) {
      double b = thresholds[i];
      if (wx < b)
        return i;
    }
    return thresholds.length;
  }

  protected double predictRelScore(Document doc) {
    double wx = 0;
    for (int i = 0; i < doc.getFeatures().size(); i++) {
      double feature = doc.getFeatures().get(i);
      wx += feature * weights[i];     //w*x
    }
    return wx;
  }

}