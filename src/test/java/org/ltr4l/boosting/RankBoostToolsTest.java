package org.ltr4l.boosting;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.*;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import static org.ltr4l.boosting.RBDistributionTest.assertRankedDocs;
import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class RankBoostToolsTest {
  private List<RankedDocs> queries;
  private List<Document> allDocs;
  private RankBoostTools rbt;

  @Before
  public void setUp() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 17.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 40.0, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    int[] labels = {0, 0, 1, 1, 2};
    addLabels(docList, labels);
    RankedDocs rDocs1 = new RankedDocs(docList);

    for(double[] document : docs)
      for(int i = 0; i < document.length; i++) {
        document[i] *= 1.1; //Chosen so that the best feature can be determined easily...
      }
    docList = makeDocsWithFeatures(docs);
    addLabels(docList, labels);
    RankedDocs rDocs2 = new RankedDocs(docList);

    for(double[] document : docs)
      for(int i = 0; i < document.length; i++)
        document[i] *= 1.1;
    docList = makeDocsWithFeatures(docs);
    addLabels(docList, labels);
    RankedDocs rDocs3 = new RankedDocs(docList);

    queries = new ArrayList<>();
    queries.add(rDocs1);
    queries.add(rDocs2);
    queries.add(rDocs3);
    allDocs = new ArrayList<>();
    queries.forEach(rd -> allDocs.addAll(rd));

    RBDistribution distribution = RBDistribution.getInitDist(queries);
    rbt = new RankBoostTools(distribution.calcPotential(), queries);
  }

  @Test
  public void testFindThresholdNoStep() throws Exception{
    //Nicely split case tested...
    FeatureSortedDocs fsd = FeatureSortedDocs.get(allDocs, 4);
    double[] threshLossq = rbt.findThreshold(fsd);
    Assert.assertEquals(17.0, threshLossq[0], 0.01);
    Assert.assertEquals(4d/3, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    //TODO: Write for other features...
  }

  @Test
  public void testFindAbsMinThreshold() throws Exception{
    OptimalLeafLoss oml = rbt.findMinLeafThreshold(allDocs, 0);
    Assert.assertEquals(17.0, oml.getOptimalThreshold(), 0.01);
    Assert.assertEquals(4d/3, oml.getMinLoss(), 0.01);
    Assert.assertEquals(4, oml.getOptimalFeature());
  }

  @Test
  public void testSearchStepThreshold() throws Exception{
    FeatureSortedDocs fsd = FeatureSortedDocs.get(allDocs, 4);

    double[] thresholds = rbt.makeStepThresholds(fsd.getMinFeature(), fsd.getMaxFeature(), 3);
    double[] threshLossq = rbt.searchStepThresholds(fsd, thresholds);
    Assert.assertEquals(16.8, threshLossq[0], 0.01);
    Assert.assertEquals(4d/3, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    thresholds = rbt.makeStepThresholds(fsd.getMinFeature(), fsd.getMaxFeature(), 2);
    threshLossq = rbt.searchStepThresholds(fsd, thresholds);
    Assert.assertEquals(24.7, threshLossq[0], 0.01);
    Assert.assertEquals(8d/5, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    thresholds = rbt.makeStepThresholds(fsd.getMinFeature(), fsd.getMaxFeature(), 4);
    threshLossq = rbt.searchStepThresholds(fsd, thresholds);
    Assert.assertEquals(12.85, threshLossq[0], 0.01);
    Assert.assertEquals(4d/3, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    thresholds = rbt.makeStepThresholds(fsd.getMinFeature(), fsd.getMaxFeature(), 5);
    threshLossq = rbt.searchStepThresholds(fsd, thresholds);
    Assert.assertEquals(10.48, threshLossq[0], 0.01);
    Assert.assertEquals(4d/3, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    thresholds = rbt.makeStepThresholds(fsd.getMinFeature(), fsd.getMaxFeature(), 6);
    threshLossq = rbt.searchStepThresholds(fsd, thresholds);
    Assert.assertEquals(8.9, threshLossq[0], 0.01);
    Assert.assertEquals(4d/3, threshLossq[1], 0.01);
    Assert.assertEquals(1.0, threshLossq[2], 0.01);

    //Default
    OptimalLeafLoss oml = rbt.findMinLeafThreshold(allDocs, 15);
    Assert.assertEquals(17.0, oml.getOptimalThreshold(), 0.01);
    Assert.assertEquals(4d/3, oml.getMinLoss(), 0.01);
    Assert.assertEquals(4, oml.getOptimalFeature());
  }

}