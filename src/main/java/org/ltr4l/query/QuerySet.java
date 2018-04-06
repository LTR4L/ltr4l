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

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class QuerySet {

  private List<Query> queries;

  public QuerySet() {
    queries = new ArrayList<>();
  }

  public void addQuery(Query query){
    queries.add(query);
  }

  public static QuerySet create(String file){
    try(InputStream is = new FileInputStream(file)){
      try(Reader reader = new InputStreamReader(is)){
        QuerySet querySet = new QuerySet();
        querySet.parseQueries(reader);
        return querySet;
      }
    }
    catch (IOException e){
      throw new RuntimeException(e);
    }
  }

  public static int findMaxLabel(List<Query> queries) {
    int maxLabel = 0;
    for (Query query : queries) {
      for (Document doc : query.getDocList()) {
        if (maxLabel < doc.getLabel())
          maxLabel = doc.getLabel();
      }
    }
    return maxLabel;
  }

  public List<Query> getQueries() {
    return queries;
  }

  public int getFeatureLength() {
    if (queries.isEmpty()) {
      System.err.println("No valid documents.");
      return -1;
    }
    return queries.get(0).getFeatureLength();
  }

  // Queries Parser---------------
  //  Dataset - LETOR 4.0
  public void parseQueries(Reader reader) throws IOException {
    BufferedReader br = new BufferedReader(reader);
    String line = "";
    while (!(line == null)) {
      line = br.readLine();
      if (!(line == null) && !line.equals(""))
        makeDocumentVector(line);
    }
  }

  //  Dataset format: svmlight / libsvm format
  //  <label> <feature-id>:<feature-value>... #docid = <feature-value> inc = <feature-value> prob = <feature-value>
  private void makeDocumentVector(String line) {
    String[] queryDocumentInfo = line.split("#docid")[0].split(" ");
    int label = Integer.parseInt(queryDocumentInfo[0]);
    int qid = Integer.parseInt(queryDocumentInfo[1].split(":")[1]); //queryid
    Document document = new Document();
    document.setLabel(label);
    for (int i = 2; i < queryDocumentInfo.length; i++) {             //Parse the line for document features
      double feature = Double.parseDouble(queryDocumentInfo[i].split(":")[1]);
      document.addFeature(feature);
    }
    boolean queryFound = false;
    if (!queries.isEmpty()) {
      for (Query query : queries) {
        if (qid == query.getQueryId()) {
          query.addDocument(document);
          queryFound = true;
          break;
        }
      }
    }
    if (queries.isEmpty() || !queryFound) {
      Query newQuery = new Query();
      newQuery.addDocument(document);
      newQuery.setQueryId(qid);
      queries.add(newQuery);
    }
  }

}

