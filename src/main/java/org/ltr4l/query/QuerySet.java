package org.ltr4l.query;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

public class QuerySet {

  private List<Query> queries;

  public QuerySet() {
    queries = new ArrayList<>();
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
      System.out.println("No valid documents.");
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

/*    public void printQueryDocumentInformation() {
        for (Query query : queries) {
            for (Document document : query.getDocList()) {
                System.out.println("qid: " + query.getQueryId() + " label: " + document.getLabel() + " features: " + document.getFeatures().toString());
            }
        }
    }*/

}

