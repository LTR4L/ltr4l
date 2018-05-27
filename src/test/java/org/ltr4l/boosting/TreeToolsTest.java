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
  public void testFindOptimalLeaf() throws Exception{
    //leaf1 has minimum split loss greater than 0.
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 10.0, 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> leaf1Docs = makeDocsWithFeatures(docs);
    leaf1Docs.get(0).setLabel(0);
    leaf1Docs.get(1).setLabel(0);
    leaf1Docs.get(2).setLabel(1);
    leaf1Docs.get(3).setLabel(1);
    leaf1Docs.get(4).setLabel(0);


    //leaf2 has minimum split loss of 0.
    docs = new double[][] {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> leaf2Docs = makeDocsWithFeatures(docs);
    leaf2Docs.get(0).setLabel(0);
    leaf2Docs.get(1).setLabel(0);
    leaf2Docs.get(2).setLabel(1);
    leaf2Docs.get(3).setLabel(1);
    leaf2Docs.get(4).setLabel(0);


    //leaf3 has minimum split loss greater than 0.
    docs = new double[][] {
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 1, 1, 1},
    };
    List<Document> leaf3Docs = makeDocsWithFeatures(docs);
    leaf3Docs.get(0).setLabel(0);
    leaf3Docs.get(1).setLabel(0);
    leaf3Docs.get(2).setLabel(1);

    Split leaf1 = new Split(null, leaf1Docs, 1);
    Split leaf2 = new Split(null, leaf2Docs, 2);
    Split leaf3 = new Split(null, leaf3Docs, 3);

    Map<Split, OptimalLeafLoss> leafLossMap = new HashMap<>();
    leafLossMap.put(leaf1, TreeTools.findMinLeafThreshold(leaf1.getScoredDocs()));
    leafLossMap.put(leaf2, TreeTools.findMinLeafThreshold(leaf2.getScoredDocs()));
    leafLossMap.put(leaf3, TreeTools.findMinLeafThreshold(leaf3.getScoredDocs()));

    Split optimalLeaf = TreeTools.findOptimalLeaf(leafLossMap);
    Assert.assertEquals(optimalLeaf, leaf2); //TODO: End test here?
    OptimalLeafLoss optLeafLoss = leafLossMap.get(optimalLeaf);

    Assert.assertEquals(optLeafLoss.getOptimalThreshold(), 10, 0.01);
    Assert.assertEquals(optLeafLoss.getOptimalFeature(), 4);
    Assert.assertEquals(optLeafLoss.getMinLoss(), 0, 0.01);
  }

  @Test
  public void testDefaultFindMinLeafThreshold() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);

    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    OptimalLeafLoss leafLoss = TreeTools.findMinLeafThreshold(docList);
    Assert.assertEquals(leafLoss.getMinLoss(), 0, 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), 4);
    Assert.assertEquals(leafLoss.getOptimalThreshold(), 10.0, 0.01);

    OptimalLeafLoss leafLoss2 = TreeTools.findMinLeafThreshold(docList, 10); //Default value
    Assert.assertEquals(leafLoss.getMinLoss(), leafLoss2.getMinLoss(), 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), leafLoss2.getOptimalFeature());
    Assert.assertEquals(leafLoss.getOptimalThreshold(), leafLoss2.getOptimalThreshold(), 0.01);

    leafLoss2 = TreeTools.findMinLeafThreshold(docList, 0); //Because greater than the number of samples
    Assert.assertEquals(leafLoss.getMinLoss(), leafLoss2.getMinLoss(), 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), leafLoss2.getOptimalFeature());
    Assert.assertEquals(leafLoss.getOptimalThreshold(), leafLoss2.getOptimalThreshold(), 0.01);
  }

  @Test
  public void testStepFindMinLeafThreshold() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);

    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    OptimalLeafLoss leafLoss = TreeTools.findMinLeafThreshold(docList, 4);
    Assert.assertEquals(leafLoss.getMinLoss(), 0, 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), 4);
    Assert.assertEquals(leafLoss.getOptimalThreshold(), 7.5225, 0.01);
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
  public void testFindThresholdStep() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    //Check that when step size is too big (numSteps too small), minimum error threshold may not be found.
    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(docList, 4);
    double[] threshLoss = TreeTools.findThreshold(sortedDocs, 2);
    Assert.assertEquals(threshLoss[0], 15.015, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 4);
    threshLoss = TreeTools.findThreshold(sortedDocs, 3);
    Assert.assertEquals(threshLoss[0], 10.02, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    //Test that it finds a candidate threshold with smaller loss.
    sortedDocs = FeatureSortedDocs.get(docList, 4);
    threshLoss = TreeTools.findThreshold(sortedDocs, 4);
    Assert.assertEquals(threshLoss[0], 7.5225, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    //Check for default.
    threshLoss = TreeTools.findThreshold(sortedDocs, 5);
    Assert.assertEquals(threshLoss[0], 10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    //Check for default.
    threshLoss = TreeTools.findThreshold(sortedDocs, 6);
    Assert.assertEquals(threshLoss[0], 10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

  }

  @Test
  public void testCalcSplitLoss() throws Exception{
    List<Document> subData = docs(1, 1, 1, 1);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 0, 0.01);

    subData = docs(1, 1, 1, 0);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 0.75, 0.01);

    subData = docs(1, 1, 0, 0);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 1.0, 0.01);

    subData = docs(1, 0, 0, 0);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 0.75, 0.01);

    subData = docs(0, 0, 0, 0);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 0, 0.01);

    subData = docs(0, 1, 1, 1, 1, 1, 2, 2, 3);
    Assert.assertEquals(TreeTools.calcSplitLoss(subData), 6, 0.01);
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


  protected static List<Document> makeDocsWithFeatures(double[][] docFeats){
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

  protected static List<Document> addLabels(List<Document> docList, int... labels){
    assert(labels.length == docList.size());
    for(int i = 0; i < docList.size(); i++)
      docList.get(i).setLabel(labels[i]);
    return docList;
  }


}