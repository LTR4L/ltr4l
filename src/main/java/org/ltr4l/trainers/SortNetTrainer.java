package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.nn.Activation;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.nn.Regularization;
import org.ltr4l.nn.SortNetMLP;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.*;

public class SortNetTrainer extends LTRTrainer {
  protected SortNetMLP smlp;
  protected double maxScore;
  protected double lrRate;
  protected double rgRate;
  double[][] targets;
  //protected List<Document[][]> trainingPairs;
  //protected List<Document[][]> validationPairs;


  SortNetTrainer(QuerySet training, QuerySet validation, Config config) {
    super(training, validation, config.getNumIterations());
    lrRate = config.getLearningRate();
    rgRate = config.getReguRate();
    maxScore = 0;
    targets = new double[][]{{1, 0}, {0, 1}};
    int featureLength = trainingSet.get(0).getFeatureLength();
    Object[][] networkShape = Arrays.copyOf(config.getNetworkShape(), config.getNetworkShape().length + 1);
    networkShape[networkShape.length - 1] = new Object[]{1, new Activation.Sigmoid()}; //one output node, but will be doubled during creation
    Optimizer.OptimizerFactory optFact = config.getOptFact();
    Regularization regularization = config.getReguFunction();
    String weightModel = config.getWeightInit();
    smlp = new SortNetMLP(featureLength, networkShape, optFact, regularization, weightModel);

/*        trainingPairs = new ArrayList<>();
        for (int i = 0; i < trainingSet.size(); i++){
            Query query = trainingSet.get(i);
            Document[][] documentPairs = query.orderDocPairs();
            trainingPairs.add(documentPairs);                   //add even if null, as placeholder for query.
        }

        validationPairs = new ArrayList<>();
        for (int i = 0; i < validationSet.size(); i++){
            Query query = validationSet.get(i);
            Document[][] documentPairs = query.orderDocPairs();
            validationPairs.add(documentPairs);
        }*/

  }

  //The following implementation is used for speed up.
  @Override
  public void train() {
    double threshold = 0.5;
    for (Query query : trainingSet) {
      List<Document> docList = query.getDocList();
      for (int i = 0; i < docList.size(); i++) {
        Document doc1 = docList.get(i);
        Document doc2 = docList.get(new Random().nextInt(docList.size()));
        //If the same document is chosen at random,
        // keep choosing until a different doc is chosen.
        while (doc1 == doc2)
          doc2 = docList.get(new Random().nextInt(docList.size()));

        double delta = doc1.getLabel() - doc2.getLabel();
        if (delta == 0) //if the label is the same, skip.
          continue;
        double prediction = smlp.predict(doc1, doc2);
        if (delta * prediction < threshold) {
          if (delta > 0)
            smlp.backProp(targets[0], new Error.SQUARE());
          else
            smlp.backProp(targets[1], new Error.SQUARE());

          smlp.updateWeights(lrRate, rgRate);
        }
      }
    }
  }

  //The below method looks over all pairs; this takes an extremely long time.
/*    @Override
    public void train() {
        double threshold = 0.5;
        for (Query query : trainingSet) {
            List<Document> docList = query.getDocList();
            for (int i = 0; i < docList.size() - 1; i++){
                Document doc1 = docList.get(i);
                for (int j = i + 1; j < docList.size(); j++) {
                    Document doc2 = docList.get(j);
                    double delta = doc1.getLabel() - doc2.getLabel();
                    if (delta != 0) {
                        double pred = smlp.predict(doc1, doc2);
                        if (delta * pred < threshold) { //Then backprop
                            if (delta > 0)
                                smlp.backProp(targets[0], new SQUARE());
                            else
                                smlp.backProp(targets[1], new SQUARE());
                            smlp.updateWeights(lrRate, rgRate);
                        }
                    }
                }
            }
        }
    }*/

  public double calculateLoss(List<Query> queries) {
    double loss = 0d;
    for (Query query : queries) {
      Document[][] pairs = query.orderDocPairs();
      if (pairs == null)
        continue;
      double queryLoss = 0d;
      for (Document[] pair : pairs) {
        double[] outputs = smlp.forwardProp(pair[0], pair[1]);
        queryLoss += new Error.SQUARE().error(outputs[0], targets[0][0]);
        queryLoss += new Error.SQUARE().error(outputs[1], targets[0][1]);
      }
      loss += queryLoss / (double) pairs.length;
    }
    return loss / (double) queries.size();
  }

  @Override
  public List<Document> sortP(Query query) {
    List<Document> ranks = new ArrayList<>(query.getDocList());
    //Reverse order to go from highest to lowest instead of lowest to highest.
    ranks.sort((docA, docB) -> Double.compare(0, smlp.predict(docA, docB)));
/*        ranks.sort(new Comparator<Document>() {
            @Override
            public int compare(Document o1, Document o2) {
                double prediction = smlp.predict(o1, o2);
                if (prediction > 0) return -1; //-1 because it should not be changed; higher is earlier
                if (prediction < 0) return 1;
                return 0;
            }
        });*/
    return ranks;
  }
}

