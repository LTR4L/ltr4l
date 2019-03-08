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
import java.util.*;
import java.util.function.Function;
import java.util.stream.Collectors;

public class QuerySet {

  private final List<Query> queries;
  private final Map<Integer, Query> queryMap; //TODO: Implement so that only one collection is necessary...

  public QuerySet() {
    queries = new ArrayList<>();
    queryMap = new HashMap<>();
  }
  public QuerySet(List<Query> queries) {
    this.queries = queries;
    queryMap = queries.stream().collect(Collectors.toMap(Query::getQueryId, Function.identity()));
  }

  public void addQuery(Query query){
    if (!queryMap.containsKey(query.getQueryId())) {
      queries.add(query);
      queryMap.put(query.getQueryId(), query);
    }
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
    final String[] queryDocumentInfo = line.split("#docid")[0].split(" ");
    final int label = Integer.parseInt(queryDocumentInfo[0]);
    final int qid = Integer.parseInt(queryDocumentInfo[1].split(":")[1]); //queryid
    Document document = new Document();
    document.setLabel(label);
    for (int i = 2; i < queryDocumentInfo.length; i++) {             //Parse the line for document features
      double feature = Double.parseDouble(queryDocumentInfo[i].split(":")[1]);
      document.addFeature(feature);
    }
    if (queryMap.containsKey(qid)) {
      Query query = queryMap.get(qid);
      query.addDocument(document);
    }
    else {
      Query query = new Query();
      query.addDocument(document);
      query.setQueryId(qid);
      this.addQuery(query);
    }
  }

}

