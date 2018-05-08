package org.ltr4l.boosting;

import org.junit.Test;

import org.junit.Assert;
import org.ltr4l.query.Document;

import java.util.ArrayList;
import java.util.List;

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
  public void testFindOptimalLeaf() throws Exception{
  }

  @Test
  public void testFindMinThreshold() throws Exception{
  }

  @Test
  public void testFindThreshold() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 1.0 , 2.0 , 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    //Will make labels solely based on feature 4.
/*    for(Document doc : docList)
      if(doc.getFeature(4) <= 3) doc.setLabel(0); else doc.setLabel(1);*/
    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(docList, 4);
    double[] threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0], 10, 0.001);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 0);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  -1.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 1);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  40.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 2);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 3);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  20.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 5);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  40.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 6);
    threshLoss = TreeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  -1.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);
  }

  @Test
  public void testCalcThresholdLoss() throws Exception{
    List<Document> subData = docs(1, 1, 1, 1);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 0, 0.01);

    subData = docs(1, 1, 1, 0);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 0.75, 0.01);

    subData = docs(1, 1, 0, 0);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 1.0, 0.01);

    subData = docs(1, 0, 0, 0);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 0.75, 0.01);

    subData = docs(0, 0, 0, 0);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 0, 0.01);

    subData = docs(0, 1, 1, 1, 1, 1, 2, 2, 3);
    Assert.assertEquals(TreeTools.calcThresholdLoss(subData), 6, 0.01);
  }

  @Test
  public void testFindMinLossFeat() throws Exception{
  }

  @Test
  public void testFindMinLossFeat1() throws Exception{
  }

  private static List<Document> makeDocsWithFeatures(double[][] docFeats){
    List<Document> docs = new ArrayList<>();
    for(double[] features : docFeats){
      Document doc = new Document();
      docs.add(doc);
      for(double feature : features) doc.addFeature(feature);
    }
    return docs;
  }

  private static List<Document> docs(int... labels){
    List<Document> documents = new ArrayList<>();
    for(int label: labels){
      Document doc = new Document();
      doc.setLabel(label);
      documents.add(doc);
    }
    return documents;
  }


}