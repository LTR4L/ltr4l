package org.ltr4l.boosting;

import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.*;

import static org.junit.Assert.*;
import static org.ltr4l.boosting.RBDistributionTest.assertRankedDocs;
import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class RankBoostToolsTest {
  private RankBoostTools rbt;

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

    List<RankedDocs> queries = new ArrayList<>();
    queries.add(rDocs1);
    queries.add(rDocs2);
    queries.add(rDocs3);
    RBDistribution distribution = RBDistribution.getInitDist(queries, 24);
    Map<Document, int[]> docMap = new HashMap<>();
    for(int qid = 0; qid < queries.size(); qid++){
      RankedDocs query = queries.get(qid);
      for(int idx = 0; idx < query.size(); idx++){
        Document key = query.get(idx);
        int[] value = {qid, idx};
        docMap.put(key, value);
      }
    }
    rbt = new RankBoostTools(WeakLearner.calculatePotential(distribution, queries), docMap);
  }

  @Test
  public void findThreshold() {
  }

  @Test
  public void searchThresholds() {
  }

  @Test
  public void calcWLloss() {
  }
}