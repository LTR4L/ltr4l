package org.ltr4l.boosting;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.ltr4l.boosting.TreeToolsTest.docs;

public class RegressionTreeToolsTest {

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
    List<Document> leaf1Docs = TreeToolsTest.makeDocsWithFeatures(docs);
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
    List<Document> leaf2Docs = TreeToolsTest.makeDocsWithFeatures(docs);
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
    List<Document> leaf3Docs = TreeToolsTest.makeDocsWithFeatures(docs);
    leaf3Docs.get(0).setLabel(0);
    leaf3Docs.get(1).setLabel(0);
    leaf3Docs.get(2).setLabel(1);

    Split leaf1 = new Split(null, leaf1Docs, 1);
    Split leaf2 = new Split(null, leaf2Docs, 2);
    Split leaf3 = new Split(null, leaf3Docs, 3);
    TreeTools treeTools = new RegressionTreeTools();

    Map<Split, OptimalLeafLoss> leafLossMap = new HashMap<>();
    leafLossMap.put(leaf1, treeTools.findMinLeafThreshold(leaf1.getScoredDocs()));
    leafLossMap.put(leaf2, treeTools.findMinLeafThreshold(leaf2.getScoredDocs()));
    leafLossMap.put(leaf3, treeTools.findMinLeafThreshold(leaf3.getScoredDocs()));

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
    List<Document> docList = TreeToolsTest.makeDocsWithFeatures(docs);

    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    TreeTools treeTools = new RegressionTreeTools();

    OptimalLeafLoss leafLoss = treeTools.findMinLeafThreshold(docList);
    Assert.assertEquals(leafLoss.getMinLoss(), 0, 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), 4);
    Assert.assertEquals(leafLoss.getOptimalThreshold(), 10.0, 0.01);

    OptimalLeafLoss leafLoss2 = treeTools.findMinLeafThreshold(docList, 10); //Default value
    Assert.assertEquals(leafLoss.getMinLoss(), leafLoss2.getMinLoss(), 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), leafLoss2.getOptimalFeature());
    Assert.assertEquals(leafLoss.getOptimalThreshold(), leafLoss2.getOptimalThreshold(), 0.01);

    leafLoss2 = treeTools.findMinLeafThreshold(docList, 0); //Because greater than the number of samples
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
    List<Document> docList = TreeToolsTest.makeDocsWithFeatures(docs);

    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    TreeTools treeTools = new RegressionTreeTools();

    OptimalLeafLoss leafLoss = treeTools.findMinLeafThreshold(docList, 4);
    Assert.assertEquals(leafLoss.getMinLoss(), 0, 0.01);
    Assert.assertEquals(leafLoss.getOptimalFeature(), 4);
    Assert.assertEquals(leafLoss.getOptimalThreshold(), 7.5225, 0.01);
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
    List<Document> docList = TreeToolsTest.makeDocsWithFeatures(docs);
    //Will make labels solely based on feature 4.
/*    for(Document doc : docList)
      if(doc.getFeature(4) <= 3) doc.setLabel(0); else doc.setLabel(1);*/
    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);
    TreeTools treeTools = new RegressionTreeTools();

    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(docList, 4);
    double[] threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0], 10, 0.001);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 0);
    threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  -1.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 1);
    threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  40.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 2);
    threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 3);
    threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  20.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 5);
    threshLoss = treeTools.findThreshold(sortedDocs);
    Assert.assertEquals(threshLoss[0],  40.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 6);
    threshLoss = treeTools.findThreshold(sortedDocs);
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
    List<Document> docList = TreeToolsTest.makeDocsWithFeatures(docs);
    docList.get(0).setLabel(0);
    docList.get(1).setLabel(0);
    docList.get(2).setLabel(1);
    docList.get(3).setLabel(1);
    docList.get(4).setLabel(0);

    TreeTools treeTools = new RegressionTreeTools();

    //Check that when step size is too big (numSteps too small), minimum error threshold may not be found.
    FeatureSortedDocs sortedDocs = FeatureSortedDocs.get(docList, 4);
    double[] threshLoss = treeTools.findThreshold(sortedDocs, 2);
    Assert.assertEquals(threshLoss[0], 15.015, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    sortedDocs = FeatureSortedDocs.get(docList, 4);
    threshLoss = treeTools.findThreshold(sortedDocs, 3);
    Assert.assertEquals(threshLoss[0], 10.02, 0.01);
    Assert.assertEquals(threshLoss[1], 0.75, 0.01);

    //Test that it finds a candidate threshold with smaller loss.
    sortedDocs = FeatureSortedDocs.get(docList, 4);
    threshLoss = treeTools.findThreshold(sortedDocs, 4);
    Assert.assertEquals(threshLoss[0], 7.5225, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    //Check for default.
    threshLoss = treeTools.findThreshold(sortedDocs, 5);
    Assert.assertEquals(threshLoss[0], 10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

    //Check for default.
    threshLoss = treeTools.findThreshold(sortedDocs, 6);
    Assert.assertEquals(threshLoss[0], 10.0, 0.01);
    Assert.assertEquals(threshLoss[1], 0, 0.01);

  }

  @Test
  public void testCalcSplitLoss() throws Exception{
    TreeTools treeTools = new RegressionTreeTools();
    List<Document> subData = docs(1, 1, 1, 1);
    Assert.assertEquals(treeTools.calcWLloss(subData), 0, 0.01);

    subData = docs(1, 1, 1, 0);
    Assert.assertEquals(treeTools.calcWLloss(subData), 0.75, 0.01);

    subData = docs(1, 1, 0, 0);
    Assert.assertEquals(treeTools.calcWLloss(subData), 1.0, 0.01);

    subData = docs(1, 0, 0, 0);
    Assert.assertEquals(treeTools.calcWLloss(subData), 0.75, 0.01);

    subData = docs(0, 0, 0, 0);
    Assert.assertEquals(treeTools.calcWLloss(subData), 0, 0.01);

    subData = docs(0, 1, 1, 1, 1, 1, 2, 2, 3);
    Assert.assertEquals(treeTools.calcWLloss(subData), 6, 0.01);
  }

}