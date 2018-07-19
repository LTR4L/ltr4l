package org.ltr4l.boosting;

import org.junit.Test;
import org.junit.Assert;

import org.ltr4l.query.Document;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class TreeToolsTest {

  @Test
  public void testOrderByFeature() throws Exception{
    double[][] docs = {
        {-1.0, 0d,1.0, 2.0, 3.0, 4.0, 5.0},          //Doc 1
        {5.0, 4.0, 3.0, 2.0, 1.0, 0d, -1.0},         //Doc 2 etc...
        {-10.0, 0d, 10.0, 20.0, 30.0, 40.0, 50.0},
        {50.0, 40.0, 30.0, 20.0, 10.0, 0d, -10.0},
        {-0.01, 0d, 0.01, 0.02, 0.03, 0.04, 0.05}
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    List<Document> sortedDocs = TreeTools.orderByFeature(docList, 0);
    Assert.assertTrue(sortedDocs.get(0) == docList.get(2));
    Assert.assertTrue(sortedDocs.get(1) == docList.get(0));
    Assert.assertTrue(sortedDocs.get(2) == docList.get(4));
    Assert.assertTrue(sortedDocs.get(3) == docList.get(1));
    Assert.assertTrue(sortedDocs.get(4) == docList.get(3));

    sortedDocs = TreeTools.orderByFeature(docList, 1);
    Assert.assertTrue(sortedDocs.get(0) == docList.get(0) || sortedDocs.get(0) == docList.get(2) || sortedDocs.get(0) == docList.get(4));
    Assert.assertTrue(sortedDocs.get(1) == docList.get(0) || sortedDocs.get(1) == docList.get(2) || sortedDocs.get(1) == docList.get(4));
    Assert.assertTrue(sortedDocs.get(2) == docList.get(0) || sortedDocs.get(2) == docList.get(2) || sortedDocs.get(2) == docList.get(4));
    Assert.assertTrue(sortedDocs.get(3) == docList.get(1));
    Assert.assertTrue(sortedDocs.get(4) == docList.get(3));

    sortedDocs = TreeTools.orderByFeature(docList, 2);
    Assert.assertTrue(sortedDocs.get(0) == docList.get(4));
    Assert.assertTrue(sortedDocs.get(1) == docList.get(0));
    Assert.assertTrue(sortedDocs.get(2) == docList.get(1));
    Assert.assertTrue(sortedDocs.get(3) == docList.get(2));
    Assert.assertTrue(sortedDocs.get(4) == docList.get(3));
  }

  @Test(expected = AssertionError.class)
  public void testOrderByFeatureOutOfRange() throws Exception{
    double[][] docs = {
        {-1.0, 0d,1.0, 2.0, 3.0, 4.0, 5.0},
        {5.0, 4.0, 3.0, 2.0, 1.0, 0d, -1.0},
        {-10.0, 0d, 10.0, 20.0, 30.0, 40.0, 50.0},
        {50.0, 40.0, 30.0, 20.0, 10.0, 0d, -10.0},
        {-0.01, 0d, 0.01, 0.02, 0.03, 0.04, 0.05}
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    List<Document> sortedDocs = TreeTools.orderByFeature(docList, 10);
  }

  @Test
  public void testFindMinLossFeat() throws Exception{
    double[][] threshLosses = {
        {0.01, 0.82},
        {1.50, 0.31},
        {0.19, 1.24},
        {5.23, 0.00},
        {92.0, 0.02},
        {100d, 0.00},
        {-5.1, 0.40}

    };

    Assert.assertEquals(TreeTools.findMinLossFeat(threshLosses), 3);
    Assert.assertEquals(TreeTools.findMinLossFeat(threshLosses, 0.01), 4);
    Assert.assertEquals(TreeTools.findMinLossFeat(threshLosses, 0.39), 6);
    Assert.assertEquals(TreeTools.findMinLossFeat(threshLosses, 1.25), -1);
  }

  @Test
  public void testBinaryThresholdSearch() throws Exception{
    //Even number of elements
    double[] featSamples = {0.01, 0.04, 1.00, 1.50, 2.20, 2.50, 2.70, 2.80, 3.00, 3.30, 3.30, 3.30, 4.89, 5.23};
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0), 0);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0.02), 1);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0.08), 2);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 1.00), 2);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 1.50), 3);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.20), 4);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.50), 5);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.60), 6);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.79), 7);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 3.00), 8);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 3.30), 9);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 4.89), 12);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 5.23), 13);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 5.24), 14);

    //Odd number of elements
    featSamples = new double[]{0.01, 0.04, 1.00, 1.50, 2.20, 2.50, 2.70, 2.80, 3.00, 3.30, 3.30};
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0), 0);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0.02), 1);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 0.08), 2);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 1.00), 2);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 1.50), 3);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.20), 4);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.50), 5);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.60), 6);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 2.79), 7);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 3.00), 8);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 3.30), 9);
    Assert.assertEquals(TreeTools.binaryThresholdSearch(featSamples, 4.89), 11);
  }


  public static List<Document> makeDocsWithFeatures(double[][] docFeats){
    List<Document> docs = new ArrayList<>();
    for(double[] features : docFeats){
      Document doc = new Document();
      docs.add(doc);
      for(double feature : features) doc.addFeature(feature);
    }
    return docs;
  }

  protected static List<Document> docs(int... labels){
    List<Document> documents = new ArrayList<>();
    for(int label: labels){
      Document doc = new Document();
      doc.setLabel(label);
      documents.add(doc);
    }
    return documents;
  }

  public static List<Document> addLabels(List<Document> docList, int... labels){
    assert(labels.length == docList.size());
    for(int i = 0; i < docList.size(); i++)
      docList.get(i).setLabel(labels[i]);
    return docList;
  }


}