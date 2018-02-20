package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;
import org.ltr4l.nn.MLP;
import org.ltr4l.nn.Optimizer;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;
import org.ltr4l.nn.Regularization;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

abstract class MLPTrainer extends LTRTrainer {
    protected MLP mlp;
    protected double maxScore;
    protected double lrRate;
    protected double rgRate;

    MLPTrainer(QuerySet training, QuerySet validation, Config config) {
        this(training, validation, config, false);
    }

    //This constructor exists solely for the purpose of child classes
    //It gives child classes the ability to assign an extended MLP.
    MLPTrainer(QuerySet training, QuerySet validation, Config config, boolean hasOtherMLP) {
        super(training, validation, config.getNumIterations());
        lrRate = config.getLearningRate();
        rgRate = config.getReguRate();
        maxScore = 0;
        if(!hasOtherMLP) {
            int featureLength = trainingSet.get(0).getFeatureLength();
            Object[][] networkShape = config.getNetworkShape();
            Optimizer.OptimizerFactory optFact = config.getOptFact();
            Regularization regularization = config.getReguFunction();
            String weightModel = config.getWeightInit();
            mlp = new MLP(featureLength, networkShape, optFact, regularization, weightModel);
        }
    }

    protected double calculateLoss(List<Query> queries){
        // Note: appears to be just as use of nested loops without streams.
        // However, I have not tested it thoroughly.
        double loss = 0d;
        for (Query query : queries) {
            List<Document> docList = query.getDocList();
            loss += docList.stream().mapToDouble(doc -> new Error.SQUARE().error(mlp.predict(doc), doc.getLabel())).sum() / docList.size();
        }
        return loss / queries.size();
    }

    @Override
    public List<Document> sortP(Query query) {
        List<Document> ranks = new ArrayList<>(query.getDocList());
        ranks.sort(Comparator.comparingDouble(mlp::predict).reversed());
        return ranks;
    }
}
