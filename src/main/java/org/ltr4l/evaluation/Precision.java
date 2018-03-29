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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

public interface Precision {

  public static double precision(List<Document> docRanks, int position){
    assert(position > -1);
    final int pos = Math.min(position, docRanks.size());
    int total = 0;
    for (int i = 0; i < pos; i++) if (docRanks.get(i).getLabel() != 0) total++;
    return ((double) total) / pos;
  }

  /**
   * Calculates Average Precision.
   * The default method, calculateAvgAllQueries(), returns MAP.
   */
  public static class AP implements RankEval {

    //Calculates Average Precision.
    public double calculate(List<Document> docRanks, int position){
      return calculate(docRanks);
    }

    //TODO: Confirm what "total number of relevant documents" means, and what to do in the case of 0.
    public double calculate(List<Document> docRanks){
      double ap = 0;
      int numRelDocs = RankEval.countNumRelDocs(docRanks);
      for (int k = 0; k < docRanks.size(); k++) {
        Document doc = docRanks.get(k);
        double rel = doc.getLabel() == 0 ? 0 : 1;
        ap += precision(docRanks, k + 1) * rel;
      }
      return ap / numRelDocs;
    }
  }


  //Weighted Average Precision.
  //Reference: https://www.nii.ac.jp/TechReports/public_html/05-014E.pdf
  static class WAP implements RankEval{

    public double calculate(List<Document> docRanks, int position){
      return calculate(docRanks);
    }
    public double calculate(List<Document> docRanks){
      List<Document> idealRanking = new ArrayList<>(docRanks);
      int numRelDocs = RankEval.countNumRelDocs(docRanks);
      idealRanking.sort(Comparator.comparingInt(Document::getLabel).reversed());
      //int pos = Math.min(position, docRanks.size());
      double total = 0d;
      for (int k = 0; k < docRanks.size(); k++){
        total += identity(docRanks.get(k)) * RankEval.cg(docRanks, k+1) / RankEval.cg(idealRanking, k+1);
      }
      return total / numRelDocs;
    }
  }
}
