package org.ltr4l.boosting;

import org.junit.Before;
import org.junit.Test;
import org.junit.Assert;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class RBDistributionTest {
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
      for(double feature : document)
        feature *= random.nextDouble();
    docList = makeDocsWithFeatures(docs);
    addLabels(docList, labels);
    RankedDocs rDocs2 = new RankedDocs(docList);
    assertRankedDocs(rDocs2);

    for(double[] document : docs)
      for(double feature : document)
        feature *= 10 * random.nextDouble();
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
  public void testGetInitDist() throws Exception{
    RBDistribution distribution = RBDistribution.getInitDist(queries, 24);
    double[][] qDist = distribution.getQueryDist(0);
    Assert.assertEquals(qDist[0][0], 0d, 0.01);
    Assert.assertEquals(qDist[0][1], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][2], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][1], 0d, 0.01);
    Assert.assertEquals(qDist[1][2], 0d, 0.01);
    Assert.assertEquals(qDist[1][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][2], 0, 0.01);
    Assert.assertEquals(qDist[2][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[3][3], 0d, 0.01);
    Assert.assertEquals(qDist[3][4], 0d, 0.01);

    qDist = distribution.getQueryDist(1);
    Assert.assertEquals(qDist[0][0], 0d, 0.01);
    Assert.assertEquals(qDist[0][1], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][2], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][1], 0d, 0.01);
    Assert.assertEquals(qDist[1][2], 0d, 0.01);
    Assert.assertEquals(qDist[1][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][2], 0, 0.01);
    Assert.assertEquals(qDist[2][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[3][3], 0d, 0.01);
    Assert.assertEquals(qDist[3][4], 0d, 0.01);

    qDist = distribution.getQueryDist(2);
    Assert.assertEquals(qDist[0][0], 0d, 0.01);
    Assert.assertEquals(qDist[0][1], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][2], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[0][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][1], 0d, 0.01);
    Assert.assertEquals(qDist[1][2], 0d, 0.01);
    Assert.assertEquals(qDist[1][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[1][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][2], 0, 0.01);
    Assert.assertEquals(qDist[2][3], 1d/24, 0.01);
    Assert.assertEquals(qDist[2][4], 1d/24, 0.01);
    Assert.assertEquals(qDist[3][3], 0d, 0.01);
    Assert.assertEquals(qDist[3][4], 0d, 0.01);

  }

  @Test
  public void testUpdate() throws Exception{
  }

  @Test
  public void testUpdateQuery() throws Exception{
    RBDistribution distribution = RBDistribution.getInitDist(queries, 24);
    WeakLearner wl = new WeakLearner(4, 10.0,2);
    int qid = 0;
    double newNormFactor = distribution.updateQuery(wl, qid, queries.get(qid));
    double[][] qDist = distribution.getQueryDist(qid);

    Assert.assertEquals(qDist[0][0], 0d, 0.01);
    Assert.assertEquals(qDist[0][1], 1d/24 * Math.exp(-2), 0.01);
    Assert.assertEquals(qDist[0][2], 1d/24 * Math.exp(-2), 0.01);
    Assert.assertEquals(qDist[0][3], 1d/24 * 1, 0.01);
    Assert.assertEquals(qDist[0][4], 1d/24 * 1, 0.01);
    Assert.assertEquals(qDist[1][1], 0d, 0.01);
    Assert.assertEquals(qDist[1][2], 0d, 0.01);
    Assert.assertEquals(qDist[1][3], 1d/24 * Math.exp(2), 0.01);
    Assert.assertEquals(qDist[1][4], 1d/24 * Math.exp(2), 0.01);
    Assert.assertEquals(qDist[2][2], 0, 0.01);
    Assert.assertEquals(qDist[2][3], 1d/24 * Math.exp(2), 0.01);
    Assert.assertEquals(qDist[2][4], 1d/24 * Math.exp(2), 0.01);
    Assert.assertEquals(qDist[3][3], 0d, 0.01);
    Assert.assertEquals(qDist[3][4], 0d, 0.01);

    double actualNormFactor = (2d/24 * Math.exp(-2)) + 2d/24 + (4d/24 * Math.exp(2));
    Assert.assertEquals(newNormFactor, actualNormFactor, 0.01);
  }

  public static void assertRankedDocs(RankedDocs rDocs) throws Exception{
    int label = rDocs.getLabel(0);
    for(Document doc : rDocs){
      assert(doc.getLabel() <= label);
      label = doc.getLabel();
    }
  }
}