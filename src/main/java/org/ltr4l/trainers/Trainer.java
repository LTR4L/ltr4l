package org.ltr4l.trainers;

import org.ltr4l.tools.Config;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.query.QuerySet;

import java.util.List;

public interface Trainer {

    void train();
    //default NDCG@10
    default void validate(int iter) {
        validate(iter, 10);

    }
    void validate(int iter, int pos);
    double[] calculateLoss();
    List<Document> sortP(Query query);
    void trainAndValidate();

    class TrainerFactory {

        public static Trainer getTrainer(String algorithm, QuerySet trainingSet, QuerySet validationSet, Config config){
            switch (algorithm.toLowerCase()){
                case "prank":
                    return new PRankTrainer(trainingSet, validationSet, config);
                case "oap":
                    return new OAPBPMTrainer(trainingSet, validationSet, config);
                case "ranknet":
                    return new RankNetTrainer(trainingSet, validationSet, config);
                case "franknet":
                    return new FRankTrainer(trainingSet, validationSet, config);
                case "lambdarank":
                    return new LambdaRankTrainer(trainingSet, validationSet, config);
                case "nnrank":
                    return new NNRankTrainer(trainingSet, validationSet, config);
                case "sortnet":
                    return new SortNetTrainer(trainingSet, validationSet, config);
                case "listnet":
                    return new ListNetTrainer(trainingSet, validationSet, config);
                default:
                    return null;
            }
        }
    }
}

