package org.ltr4l.boosting;

import org.junit.Assert;
import org.junit.Test;
import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;
import static org.ltr4l.boosting.TreeToolsTest.addLabels;
import static org.ltr4l.boosting.TreeToolsTest.makeDocsWithFeatures;

public class WeakLearnerTest {

  @Test
  public void findWeakLearner() {
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

    List<RankedDocs> queries = new ArrayList<>();
    queries.add(rDocs1);
    queries.add(rDocs2);
    queries.add(rDocs3);

    RBDistribution distribution = new RBDistribution(queries);
    WeakLearner wl = WeakLearner.findWeakLearner(distribution, queries, 6);
    Assert.assertEquals(4, wl.getFid());
    Assert.assertEquals(0.5 * Math.log(7), wl.getAlpha(), 0.01);

    for(RankedDocs query : queries)
      for(int idx = 0; idx < query.size(); idx++)
        Assert.assertEquals(idx < 3 ? 1d : 0d, wl.predict(query.get(idx).getFeatures()), 0.01);
  }

}