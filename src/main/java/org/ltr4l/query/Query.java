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

package org.ltr4l.query;

import java.util.ArrayList;
import java.util.List;

public class Query {
  private List<Document> docList;
  private int queryId;
  private int featureLength;

  public Query() {
    docList = new ArrayList<>();
    queryId = -1;
    featureLength = 0;
  }

  public Query(List<Document> docList) {
    this(docList, -1);
  }

  public Query(List<Document> docList, int queryId) {
    this.docList = docList;
    this.queryId = queryId;
    featureLength = docList.get(0).getFeatureLength();
  }

  //This returns all document pairs (docA, docB), ordered such that
  // labelA >= labelB
/*    public Document[][] orderAllDocPairs () {
        int numDocs = docList.size();
        int numPairs = (numDocs) * (numDocs - 1) / 2;
        Document [][] allPairs = new Document [numPairs][2];
        // {
        // {Doc A, Doc B}
        // {Doc A, Doc C}
        // {Doc B, Doc C}
        // }
        int pairCount = 0;
        for (int i = 0; i < numDocs; i++) {
            Document docA = docList.get(i); //First document of the pair
            for (int k = i + 1; k < numDocs; k++) { //get all the possible second documents
                Document docB = docList.get(k);
                if (docA.getLabel() >= docB.getLabel()) //To ensure cover equal case...
                    allPairs[pairCount] = new Document[] {docA, docB};
                else
                    allPairs[pairCount] = new Document[] {docB, docA};
                pairCount++;
            }
        }
        return allPairs;
    }*/

  //This returns an array of document pairs (docA, docB) which strictly have labels
  // labelA > labelB. Thus pairs of labelA = labelB will not be included.
  // If the query only contains pairs with equal labels, null will be returned.
  public Document[][] orderDocPairs() {
    List<Document[]> docPairsList = new ArrayList<>();

    int numDocs = docList.size();
    //int numPairs = (numDocs) * (numDocs - 1) / 2;
    //Document [][] allPairs = new Document [numPairs][2];
    // {
    // {Doc A, Doc B}
    // {Doc A, Doc C}
    // {Doc B, Doc C}
    // }
    //int pairCount = 0;
    for (int i = 0; i < numDocs; i++) {
      Document docA = docList.get(i); //First document of the pair
      for (int k = i + 1; k < numDocs; k++) { //get all the possible second documents
        Document docB = docList.get(k);
        if (docA.getLabel() == docB.getLabel())
          continue; //if the labels are even, don't add them...
        if (docA.getLabel() > docB.getLabel()) //To ensure cover equal case...
          docPairsList.add(new Document[]{docA, docB});
        else
          docPairsList.add(new Document[]{docB, docA});
        //pairCount++;
      }
    }
    if (docPairsList.isEmpty())
      return null;
    return docPairsList.toArray(new Document[docPairsList.size()][2]);
  }

  public List<Document> getDocList() {
    return docList;
  }

  public void setQueryId(int qid) {
    queryId = qid;
  }

  public int getQueryId() {
    return queryId;
  }

  public int getFeatureLength() {
    return featureLength;
  }

  public void addDocument(Document document) {
    if (docList.isEmpty())
      featureLength = document.getFeatures().size();
    docList.add(document);
  }
}
