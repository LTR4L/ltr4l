/*
 * Copyright 2018 org.LTR4L
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ltr4l.evaluation;

import org.ltr4l.query.Document;

import java.util.Comparator;
import java.util.List;

public class DCG implements RankEval {

  /**
   * Calculate Discounted Cumulative Gain. DCG is an evaluation measure that can leverage the relevance judgement
   * in terms of multiple ordered categories, and has an explicit position discount factor in its definition.
   * @param docsRanks
   * @param position the k-position of DCG@k
   * @return the score of DCG
   */
  static double dcg(List<Document> docsRanks, int position) {
    double sum = 0;
    if (position > -1) {
      final int pos = Math.min(position, docsRanks.size());
      for (int i = 0; i < pos; i++) {
        sum += (Math.pow(2, docsRanks.get(i).getLabel()) - 1) / Math.log(i + 2);
      }
    }
    return sum * Math.log(2);  //Change of base
  }

  @Override
  public double calculate(List<Document> docRanks, int position){
    return dcg(docRanks, position);
  }

  public static class NDCG extends DCG{
    /**
     * Calculate Normalized Discounted Cumulative Gain. This is calculated by normalizing {@link DCG#dcg(List, int)}
     * {@literal @}k with its maximum possible value.
     * @param docsRanks
     * @param position the k-position of NDCG@k
     * @return the score of NDCG
     */
    public double calculate(List<Document> docsRanks, int position) {
      //Accept docs in predicted ranking order
      double dcg = dcg(docsRanks, position);
      //System.out.println(dcg);
      //Sort for ideal
      docsRanks.sort(Comparator.comparingInt(Document::getLabel).reversed()); //to arrange in order of highest to lowest
      double idealDcg = dcg(docsRanks, position);
      return idealDcg == 0 ? 0.0 :
          dcg / idealDcg;
    }
  }
}
