package org.ltr4l.trainers;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;
import org.ltr4l.tools.DataProcessor;

import java.util.ArrayList;
import java.util.List;

public class LambdaMartTrainerTest {

  @Test
  public void testMakeDocList() throws Exception{
    List<Query> queries = createQueryList(3, 2, 3);
    List<Document> docs = DataProcessor.makeDocList(queries);

    Assert.assertEquals(6, docs.size());
    Assert.assertEquals(0, docs.get(0).getLabel());
    Assert.assertEquals(1, docs.get(1).getLabel());
    Assert.assertEquals(1, docs.get(2).getLabel());
    Assert.assertEquals(2, docs.get(3).getLabel());
    Assert.assertEquals(2, docs.get(4).getLabel());
    Assert.assertEquals(3, docs.get(5).getLabel());

    Assert.assertEquals(0, (double) docs.get(0).getFeature(0), 0.01);
    Assert.assertEquals(1, (double) docs.get(0).getFeature(1), 0.01);
    Assert.assertEquals(2, (double) docs.get(0).getFeature(2), 0.01);
    Assert.assertEquals(1, (double) docs.get(1).getFeature(0), 0.01);
    Assert.assertEquals(2, (double) docs.get(1).getFeature(1), 0.01);
    Assert.assertEquals(3, (double) docs.get(1).getFeature(2), 0.01);
    Assert.assertEquals(1, (double) docs.get(2).getFeature(0), 0.01);
    Assert.assertEquals(2, (double) docs.get(2).getFeature(1), 0.01);
    Assert.assertEquals(3, (double) docs.get(2).getFeature(2), 0.01);
    Assert.assertEquals(2, (double) docs.get(3).getFeature(0), 0.01);
    Assert.assertEquals(3, (double) docs.get(3).getFeature(1), 0.01);
    Assert.assertEquals(4, (double) docs.get(3).getFeature(2), 0.01);
    Assert.assertEquals(2, (double) docs.get(4).getFeature(0), 0.01);
    Assert.assertEquals(3, (double) docs.get(4).getFeature(1), 0.01);
    Assert.assertEquals(4, (double) docs.get(4).getFeature(2), 0.01);
    Assert.assertEquals(3, (double) docs.get(5).getFeature(0), 0.01);
    Assert.assertEquals(4, (double) docs.get(5).getFeature(1), 0.01);
    Assert.assertEquals(5, (double) docs.get(5).getFeature(2), 0.01);
  }

  static Document createDoc( int docNum, int numFeatures){
    Document doc = new Document();
    doc.setLabel(docNum);

    for (int j = 0; j < numFeatures; j++){
      doc.addFeature(docNum + j);
    }
    return doc;
  }

  static List<Query> createQueryList(int numQueries, int numDocs, int numFeatures){
    List<Query> queries = new ArrayList<>();
    for (int i = 0; i < numQueries; i++){
      Query query = new Query();
      for (int j = 0; j < numDocs; j++){
        query.addDocument(createDoc(i + j, numFeatures));
      }
      queries.add(query);
    }
    return queries;
  }

}