package org.ltr4l.boosting;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;
import org.ltr4l.query.Document;

import java.io.Reader;
import java.io.StringReader;
import java.io.StringWriter;
import java.io.Writer;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ltr4l.boosting.RegressionTree.*;

public class RegressionTreeTest {
  private static final String MODEL_SRC = "{\n" +
      "    \"config\" : {\n" +
      "    \"algorithm\" : \"LambdaMart\",\n" +
      "    \"numIterations\" : 100,\n" +
      "    \"batchSize\" : 0,\n" +
      "    \"verbose\" : true,\n" +
      "    \"nomodel\" : false,\n" +
      "    \"params\" : {\n" +
      "      \"numTrees\" : 15,\n" +
      "      \"numLeaves\" : 3,\n" +
      "      \"learningRate\" : 0.05,\n" +
      "      \"regularization\" : {\n" +
      "        \"regularizer\" : \"L2\",\n" +
      "        \"rate\" : 0.01\n" +
      "      }\n" +
      "    },\n" +
      "    \"dataSet\" : {\n" +
      "      \"training\" : \"data/MQ2008/Fold1/train.txt\",\n" +
      "      \"validation\" : \"data/MQ2008/Fold1/vali.txt\",\n" +
      "      \"test\" : \"data/MQ2008/Fold1/test.txt\"\n" +
      "    },\n" +
      "    \"model\" : {\n" +
      "      \"format\" : \"json\",\n" +
      "      \"file\" : \"model/lambdamart-model.json\"\n" +
      "    },\n" +
      "    \"evaluation\" : {\n" +
      "      \"evaluator\" : \"NDCG\",\n" +
      "      \"params\" : {\n" +
      "        \"k\" : 10\n" +
      "      }\n" +
      "    },\n" +
      "    \"report\" : {\n" +
      "      \"format\" : \"csv\",\n" +
      "      \"file\" : \"report/lambdamart-report.csv\"\n" +
      "    }\n" +
      "  },\n" +
      "    \"leafIds\" : [ 0, 1, 3, 4, 2 ],\n" +
      "    \"featureIds\" : [ 1, 2, -1, -1, -1 ],\n" +
      "    \"thresh\" : [ 0.748092, 0.523628, \"-Infinity\", \"-Infinity\", \"-Infinity\" ],\n" +
      "    \"scores\" : [ 0.0, 0.0, 0.04655721205243154, -0.026720292627365028, 0.045137089557086 ]\n" +
      "  }";

  private RegressionTree tree;
  private RegressionTree.SavedModel model;

  @Before
  public void setUp() throws Exception{
    Reader reader = new StringReader(MODEL_SRC);
    ObjectMapper mapper = new ObjectMapper();
    model = mapper.readValue(reader, RegressionTree.SavedModel.class);
    tree = new RegressionTree(model);
  }

  @Test
  public void testReadConstruction() throws Exception{
    Split leaf = tree.getRoot();
    Assert.assertTrue(leaf.isRoot());
    Assert.assertTrue(leaf.hasDestinations());
    Assert.assertTrue(leaf.leavesProperlySet());
    Assert.assertTrue(!leaf.getRightLeaf().hasDestinations());
    Assert.assertEquals(leaf.getLeafId(), 0);

    Assert.assertTrue(!leaf.getRightLeaf().hasDestinations());
    Assert.assertEquals(leaf.getRightLeaf().getLeafId(), 2);
  }


  @Test
  public void getModelInfo() throws Exception{
    List<Double> scores = tree.getModelInfo(RegressionTree.DoubleProp.SCORE);
    List<Double> thresholds = tree.getModelInfo(RegressionTree.DoubleProp.THRESHOLD);
    List<Integer> featIds = tree.getModelInfo(RegressionTree.IntProp.FEATURE);
    List<Integer> leafIds = tree.getModelInfo(RegressionTree.IntProp.ID);

    Assert.assertEquals(scores.size(), 5);
    Assert.assertEquals(thresholds.size(), 5);
    Assert.assertEquals(featIds.size(), 5);
    Assert.assertEquals(leafIds.size(), 5);

    Assert.assertEquals(scores.get(0), 0.0, 0.01);
    Assert.assertEquals(scores.get(1), 0.0, 0.01);
    Assert.assertEquals(scores.get(2), 0.04655721205243154, 0.01);
    Assert.assertEquals(scores.get(3), -0.026720292627365028, 0.01);
    Assert.assertEquals(scores.get(4), 0.045137089557086, 0.01);

    Assert.assertEquals(thresholds.get(0),0.748092, 0.01 );
    Assert.assertEquals(thresholds.get(1),0.523628, 0.01 );
    Assert.assertEquals(thresholds.get(2),Double.NEGATIVE_INFINITY, 0.01 );
    Assert.assertEquals(thresholds.get(3),Double.NEGATIVE_INFINITY, 0.01 );
    Assert.assertEquals(thresholds.get(4),Double.NEGATIVE_INFINITY, 0.01 );

    Assert.assertEquals((int) featIds.get(0), 1);
    Assert.assertEquals((int) featIds.get(1), 2);
    Assert.assertEquals((int) featIds.get(2), -1);
    Assert.assertEquals((int) featIds.get(3), -1);
    Assert.assertEquals((int) featIds.get(4), -1);

    Assert.assertEquals((int) leafIds.get(0), 0);
    Assert.assertEquals((int) leafIds.get(1), 1);
    Assert.assertEquals((int) leafIds.get(2), 3);
    Assert.assertEquals((int) leafIds.get(3), 4);
    Assert.assertEquals((int) leafIds.get(4), 2);
  }

  @Test
  public void testGetTerminalLeaves() throws Exception{
    List<Split> terminalLeaves = tree.getTerminalLeaves();
    Assert.assertEquals(terminalLeaves.size(), model.config.getNumLeaves());
    for(Split leaf : terminalLeaves){ //Should be true in general.
      Assert.assertTrue(!leaf.hasDestinations());
      Assert.assertTrue(!leaf.isRoot());
    }
    Assert.assertEquals(terminalLeaves.get(0), tree.getRoot().getLeftLeaf().getLeftLeaf() ); //For MODEL_SRC
    Assert.assertEquals(terminalLeaves.get(1), tree.getRoot().getLeftLeaf().getRightLeaf());
    Assert.assertEquals(terminalLeaves.get(2), tree.getRoot().getRightLeaf());
  }

  @Test
  public void predict() throws Exception{
    double[] features1 = {5290, 0.748091, 0.523627};
    double[] features2 = {1609, 0.748091, 0.523628};
    double[] features3 = {1234, 0.748092, 0.523627};
    double[] features4 = {1337, 0.748092, 0.523628};

    List<Double> features = Arrays.stream(features1).boxed().collect(Collectors.toList());
    Assert.assertEquals(tree.predict(features), 0.04656, 0.00001);

    features = Arrays.stream(features2).boxed().collect(Collectors.toList());
    Assert.assertEquals(tree.predict(features), -0.02672, 0.00001);

    features = Arrays.stream(features3).boxed().collect(Collectors.toList());
    Assert.assertEquals(tree.predict(features), 0.04513, 0.00001);

    features = Arrays.stream(features4).boxed().collect(Collectors.toList());
    Assert.assertEquals(tree.predict(features), 0.04513, 0.00001);
  }

  @Test
  public void testWriteModel() throws Exception{
    Ensemble.TreeConfig config = model.config;
    Writer writer = new StringWriter();
    tree.writeModel(config, writer);
    String output = writer.toString();
    output.equals(MODEL_SRC);
  }

  @Test(expected = InvalidFeatureThresholdException.class) //TODO: Add more tests to test this exception
  public void testInvalidInitialSplit() throws Exception{
    double[][] features = {
        {3, 10, 100},
        {2, 20, 200},
        {1, 30, 400},
        {6, 35, 360}
    };
    List<Document> samples = TreeToolsTest.makeDocsWithFeatures(features);
    RegressionTree tree = new RegressionTree(3, 1, 40, samples);
  }
}