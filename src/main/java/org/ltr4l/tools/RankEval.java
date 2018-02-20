package org.ltr4l.tools;

import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.trainers.Trainer;

import java.util.Comparator;
import java.util.List;

public class RankEval {

  private static double dcg(List<Document> docsRanks, int position) {
    double sum = 0;
    if (position > -1) {
      final int pos = Math.min(position, docsRanks.size());
      for (int i = 0; i < pos; i++) {
        sum += (Math.pow(2, docsRanks.get(i).getLabel()) - 1) / Math.log(i + 2);
      }
    }
    return sum * Math.log(2);  //Change of base
  }

  private static double ndcg(List<Document> docsRanks, int position) {
    //Accept docs in predicted ranking order
    double dcg = dcg(docsRanks, position);
    //System.out.println(dcg);
    //Sort for ideal
    docsRanks.sort(Comparator.comparingInt(Document::getLabel).reversed()); //to arrange in order of highest to lowest
    double idealDcg = dcg(docsRanks, position);
    return idealDcg == 0 ? 0.0 :
        dcg / idealDcg;
  }

  public static double ndcgAvg(Trainer trainer, List<Query> queries, int position) {
    double total = 0;
    for (Query query : queries) {
      total += ndcg(trainer.sortP(query), position);
    }
    return total / queries.size();
  }

  //The method commented out is to compensate for "comparing method violates its general contract"
/*    public static double ndcgAvg(Trainer trainer, List<Query> queries, int position){
        double total = 0;
        int badQ = 0;
        for (Query query : queries){
            //total += ndcg(trainer.sortP(query), position);
            try{
                total += ndcg(trainer.sortP(query), position);
            } catch (IllegalArgumentException e) {
                System.out.println("Comparing method violates its general contract!");
                badQ++;
            }

        }
        return total/(queries.size() - badQ);
    }*/

}
