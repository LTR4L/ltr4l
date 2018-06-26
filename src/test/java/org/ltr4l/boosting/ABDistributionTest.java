package org.ltr4l.boosting;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.ltr4l.boosting.RBDistributionTest.assertRankedDocs;
import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class ABDistributionTest {
  private List<RankedDocs> queries;

  @Before
  public void setUp() throws Exception{
    double[][] docs = {
        {-1.0 , 0d  , 20.0, 20.0, 3.0 , 4.0 , 5.0  },
        {5.0  , 4.0 , 3.0 , 2.0 , 1.0 , 0d  , -1.0 },
        {-10.0, 0d  , 10.0, 20.0, 30.0, 40.0, 50.0 },
        {50.0 , 40.0, 30.0, 20.0, 10.0, 0d  , -10.0},
        {-0.01, 0d  , 0.01, 0.02, 0.03, 0.04, 0.05 }
    };
    List<Document> docList = makeDocsWithFeatures(docs);
    int[] labels = {0, 0, 1, 1, 2};
    addLabels(docList, labels);
    RankedDocs rDocs1 = new RankedDocs(docList);
    assertRankedDocs(rDocs1);

    Random random = new Random();

    for(double[] document : docs)
      for(int i = 0; i < document.length; i++)
        document[i] *= random.nextDouble();
    docList = makeDocsWithFeatures(docs);
    addLabels(docList, labels);
    RankedDocs rDocs2 = new RankedDocs(docList);
    assertRankedDocs(rDocs2);

    for(double[] document : docs)
      for(int i = 0; i < document.length; i++)
        document[i] *= 10 * random.nextDouble();
    docList = makeDocsWithFeatures(docs);
    addLabels(docList, labels);
    RankedDocs rDocs3 = new RankedDocs(docList);
    assertRankedDocs(rDocs3);

    queries = new ArrayList<>();
    queries.add(rDocs1);
    queries.add(rDocs2);
    queries.add(rDocs3);
  }

  @Test
  public void testInitialize() throws Exception {
    ABDistribution ABDistribution = new ABDistribution(queries);
    double[][] dist = ABDistribution.getFullDist();
    for(double[] qDist : dist)
      for(double d : qDist)
        Assert.assertEquals(1d/15, d, 0.01);
  }

  @Test
  public void testUpdate() throws Exception{
  }

  @Test
  public void testUpdateQuery() {
    ABDistribution ABDistribution = new ABDistribution(queries);
    WeakLearner wl = new AdaWeakLearner(4, 10.0,2);
    int qid = 0;
    double newNormFactor = ABDistribution.updateQuery(wl, qid, queries.get(qid).getRankedDocs());
    double[] qDist = ABDistribution.getQueryDist(qid);
    Assert.assertEquals(Math.exp(2)/15, qDist[0], 0.000001);
    Assert.assertEquals(Math.exp(-2)/15, qDist[1], 0.000001);
    Assert.assertEquals(Math.exp(-2)/15, qDist[2], 0.000001);
    Assert.assertEquals(Math.exp(-2)/15, qDist[3], 0.000001);
    double actualNormFactor = (3d/15 * Math.exp(-2)) + 1d/15 * Math.exp(2);
    Assert.assertEquals(newNormFactor, actualNormFactor, 0.01);
  }

}