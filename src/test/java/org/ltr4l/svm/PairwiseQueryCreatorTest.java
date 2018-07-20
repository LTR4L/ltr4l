package org.ltr4l.svm;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.boosting.TreeToolsTest;
import org.ltr4l.query.Document;
import org.ltr4l.query.Query;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

public class PairwiseQueryCreatorTest {
  private Query A;
  private Query B;
  private Query C;
  private List<Query> queries;

  @Before
  public void setUp() throws Exception {
    double[][] docs = {
        {-1.0, 0d, 1.0},          //Doc 1
        {5.0, 4.0, 3.0},         //Doc 2 etc...
        {-10.0, 0d, 10.0},
        {50.0, 40.0, 30.0},
    };
    List<Document> docsA = TreeToolsTest.makeDocsWithFeatures(docs);
    TreeToolsTest.addLabels(docsA, 2, 1, 0, 0);
    for (double[] doc : docs)
      for (int i = 0; i < doc.length; i++)
        doc[i] += 2.34;
    List<Document> docsB = TreeToolsTest.makeDocsWithFeatures(docs);
    TreeToolsTest.addLabels(docsB, 1, 1, 0, 0);
    for (double[] doc : docs)
      for (int i = 0; i < doc.length; i++)
        doc[i] -= 2d;
    List<Document> docsC = TreeToolsTest.makeDocsWithFeatures(docs);
    TreeToolsTest.addLabels(docsC, 1, 0, 0, 0);
    A = new Query(docsA);
    B = new Query(docsB);
    C = new Query(docsC);
    queries = new ArrayList<>();
    queries.add(A);
    queries.add(B);
    queries.add(C);
  }

  @Test
  public void testCreateQueries() throws Exception {
    List<Query> pwQueries = PairwiseQueryCreator.createQueries(queries);
    Assert.assertEquals(3, pwQueries.size());
    List<Document> qDocs = pwQueries.get(0).getDocList();
    Assert.assertEquals(10, qDocs.size());

    Document doc = qDocs.get(0);
    Assert.assertEquals(-6d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-2d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(1);
    Assert.assertEquals(6d, doc.getFeature(0), 0.01);
    Assert.assertEquals(4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(2d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(2);
    Assert.assertEquals(9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(3);
    Assert.assertEquals(-9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(4);
    Assert.assertEquals(-51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(5);
    Assert.assertEquals(51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(6);
    Assert.assertEquals(15d, doc.getFeature(0), 0.01);
    Assert.assertEquals(4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-7d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(7);
    Assert.assertEquals(-15d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(7d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(8);
    Assert.assertEquals(-45d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-36d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-27d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(9);
    Assert.assertEquals(45d, doc.getFeature(0), 0.01);
    Assert.assertEquals(36d, doc.getFeature(1), 0.01);
    Assert.assertEquals(27d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    qDocs = pwQueries.get(1).getDocList();
    Assert.assertEquals(8, qDocs.size());

    doc = qDocs.get(0);
    Assert.assertEquals(9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(1);
    Assert.assertEquals(-9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(2);
    Assert.assertEquals(-51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(3);
    Assert.assertEquals(51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(4);
    Assert.assertEquals(15d, doc.getFeature(0), 0.01);
    Assert.assertEquals(4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-7d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(5);
    Assert.assertEquals(-15d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(7d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(6);
    Assert.assertEquals(-45d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-36d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-27d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(7);
    Assert.assertEquals(45d, doc.getFeature(0), 0.01);
    Assert.assertEquals(36d, doc.getFeature(1), 0.01);
    Assert.assertEquals(27d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    qDocs = pwQueries.get(2).getDocList();
    Assert.assertEquals(6, qDocs.size());

    doc = qDocs.get(0);
    Assert.assertEquals(-6d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-2d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(1);
    Assert.assertEquals(6d, doc.getFeature(0), 0.01);
    Assert.assertEquals(4d, doc.getFeature(1), 0.01);
    Assert.assertEquals(2d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(2);
    Assert.assertEquals(9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(3);
    Assert.assertEquals(-9d, doc.getFeature(0), 0.01);
    Assert.assertEquals(0d, doc.getFeature(1), 0.01);
    Assert.assertEquals(9d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());

    doc = qDocs.get(4);
    Assert.assertEquals(-51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(-40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(-29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(1, doc.getLabel());

    doc = qDocs.get(5);
    Assert.assertEquals(51d, doc.getFeature(0), 0.01);
    Assert.assertEquals(40d, doc.getFeature(1), 0.01);
    Assert.assertEquals(29d, doc.getFeature(2), 0.01);
    Assert.assertEquals(-1, doc.getLabel());
  }
}