package org.ltr4l.tools;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class DataProcessorTest {
  private List<Document> docList;

  @Before
  public void setUp(){
    Document doc1 = docWithFeats(1.0, 2.0, 3.0, 4.0);
    Document doc2 = docWithFeats(2.0, 3.0, 4.0, 5.0);
    Document doc3 = docWithFeats(3.0, 4.0, 5.0, 6.0);
    Document doc4 = docWithFeats(4.0, 5.0, 6.0, 7.0);
    docList = makeDocList(doc1, doc2, doc3, doc4);
  }


  @Test
  public void testFindMinMax() throws Exception{
    double[][] featMinMax = DataProcessor.getFeatureMinMax(docList);
    Assert.assertEquals(1.0, featMinMax[0][0], 0.001);
    Assert.assertEquals(2.0, featMinMax[1][0], 0.001);
    Assert.assertEquals(3.0, featMinMax[2][0], 0.001);
    Assert.assertEquals(4.0, featMinMax[3][0], 0.001);

    Assert.assertEquals(4.0, featMinMax[0][1], 0.001);
    Assert.assertEquals(5.0, featMinMax[1][1], 0.001);
    Assert.assertEquals(6.0, featMinMax[2][1], 0.001);
    Assert.assertEquals(7.0, featMinMax[3][1], 0.001);
  }

  @Test
  public void testScale() throws Exception{
    DataProcessor.scale(docList);

    Document doc1 = docList.get(0);
    Document doc2 = docList.get(1);
    Document doc3 = docList.get(2);
    Document doc4 = docList.get(3);

    Assert.assertEquals(0d, getFeature(doc1, 0), 0.001);
    Assert.assertEquals(0d, getFeature(doc1, 1), 0.001);
    Assert.assertEquals(0d, getFeature(doc1, 2), 0.001);
    Assert.assertEquals(0d, getFeature(doc1, 3), 0.001);

    Assert.assertEquals(0.3333, getFeature(doc2, 0), 0.001);
    Assert.assertEquals(0.3333, getFeature(doc2, 0), 0.001);
    Assert.assertEquals(0.3333, getFeature(doc2, 0), 0.001);
    Assert.assertEquals(0.3333, getFeature(doc2, 0), 0.001);

    Assert.assertEquals(0.6666, getFeature(doc3, 0), 0.001);
    Assert.assertEquals(0.6666, getFeature(doc3, 0), 0.001);
    Assert.assertEquals(0.6666, getFeature(doc3, 0), 0.001);
    Assert.assertEquals(0.6666, getFeature(doc3, 0), 0.001);

    Assert.assertEquals(1, getFeature(doc4, 0), 0.001);
    Assert.assertEquals(1, getFeature(doc4, 0), 0.001);
    Assert.assertEquals(1, getFeature(doc4, 0), 0.001);
    Assert.assertEquals(1, getFeature(doc4, 0), 0.001);
  }

  @Test
  public void testGetAvgOfFeature() throws Exception{
    Assert.assertEquals(2.5, DataProcessor.getAvgOfFeature(docList, 0), 0.001);
    Assert.assertEquals(3.5, DataProcessor.getAvgOfFeature(docList, 1), 0.001);
    Assert.assertEquals(4.5, DataProcessor.getAvgOfFeature(docList, 2), 0.001);
    Assert.assertEquals(5.5, DataProcessor.getAvgOfFeature(docList, 3), 0.001);
  }

  @Test
  public void testCalcVariances() throws Exception{
    Map<Integer, Double> variances = DataProcessor.calcVariances(docList);

    Assert.assertEquals(1.25, variances.get(0), 0.001);
    Assert.assertEquals(1.25, variances.get(1), 0.001);
    Assert.assertEquals(1.25, variances.get(2), 0.001);
    Assert.assertEquals(1.25, variances.get(3), 0.001);
  }

  @Test
  public void testOrderSelectedFeatures() throws Exception{
    Document doc1 = docWithFeats(1.0, 1.0, 1.0, 1.0);
    Document doc2 = docWithFeats(2.0, 2.0, 3.0, 4.0);
    Document doc3 = docWithFeats(3.0, 4.0, 9.0, 16.0);
    Document doc4 = docWithFeats(4.0, 6.0, 27.0, 48.0);
    List<Document> docs = makeDocList(doc1, doc2, doc3, doc4);
    Map<Integer, Double> variance = DataProcessor.calcVariances(docs);

    List<Integer> orderedFeatures = DataProcessor.orderSelectedFeatures(variance, 0);
    Assert.assertEquals(4, orderedFeatures.size());
    Assert.assertEquals(3, (int) orderedFeatures.get(0));
    Assert.assertEquals(2, (int) orderedFeatures.get(1));
    Assert.assertEquals(1, (int) orderedFeatures.get(2));
    Assert.assertEquals(0, (int) orderedFeatures.get(3));

    orderedFeatures = DataProcessor.orderSelectedFeatures(variance, 1.25);
    Assert.assertEquals(3, orderedFeatures.size());
    Assert.assertEquals(3, (int) orderedFeatures.get(0));
    Assert.assertEquals(2, (int) orderedFeatures.get(1));
    Assert.assertEquals(1, (int) orderedFeatures.get(2));

    orderedFeatures = DataProcessor.orderSelectedFeatures(variance, 3.69);
    Assert.assertEquals(2, orderedFeatures.size());
    Assert.assertEquals(3, (int) orderedFeatures.get(0));
    Assert.assertEquals(2, (int) orderedFeatures.get(1));

    orderedFeatures = DataProcessor.orderSelectedFeatures(variance, 105);
    Assert.assertEquals(1, orderedFeatures.size());
    Assert.assertEquals(3, (int) orderedFeatures.get(0));

    orderedFeatures = DataProcessor.orderSelectedFeatures(variance, 350);
    Assert.assertTrue(orderedFeatures.isEmpty());
  }

  private static Document docWithFeats(double... features){
    Document doc = new Document();
    for (double feature : features) doc.addFeature(feature);
    return doc;
  }

  private static List<Document> makeDocList(Document... docs){
    List<Document> documentList = new ArrayList<>();
    for (Document doc : docs) documentList.add(doc);
    return documentList;
  }

  private static double getFeature(Document doc, int feature){
    return doc.getFeature(feature);
  }

}