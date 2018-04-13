package org.ltr4l.boosting;

import org.junit.Assert;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class TreeTest {

  @Test
  public void testTree() throws Exception{
    List<Double> features = new ArrayList<>();
    features.add(0.09);
    features.add(0.11);
    Tree tree1 = new Tree(1, 0.1 , 0.01, 0.02);
    Tree tree2 = new Tree(0, 0.1, 0.01, 0.02);

    Assert.assertEquals(0.02, tree1.score(features), 0.001);
    Assert.assertEquals(0.01, tree2.score(features), 0.001);

    tree1.getDestinationLeaf(0).setDestinationTree(tree2);
    Assert.assertEquals(0.02, tree1.score(features), 0.001);
    Assert.assertEquals(0.01, tree2.score(features), 0.001);
    resetLeafDestination(tree1, 0);

    tree1.getDestinationLeaf(1).setDestinationTree(tree2);
    Assert.assertEquals(0.01, tree1.score(features), 0.001);
    Assert.assertEquals(0.01, tree2.score(features), 0.001);

    resetLeafDestination(tree1, 1);
    tree2.getDestinationLeaf(0).setDestinationTree(tree1);
    Assert.assertTrue(tree2.getDestinationLeaf(0).getDestinationTree() == tree1);
    Assert.assertTrue(tree1.getSourceLeaf() == tree2.getDestinationLeaf(0));
    Assert.assertEquals(0.02, tree1.score(features), 0.001);
    Assert.assertEquals(0.02, tree2.score(features), 0.001);
  }

  @Test (expected = AssertionError.class)
  public void testInfiniteTree() throws Exception{
    Tree tree1 = new Tree(1, 0.1 , 0.01, 0.02);
    Tree tree2 = new Tree(0, 0.1, 0.01, 0.02);
    setLeafDestination(tree1, tree2, 1);
    setLeafDestination(tree2, tree1, 0); //Assertions.assertThrows JUnit5
  }

  @Test
  public void testIsLinkedTo() throws Exception{
    Tree tree1 = new Tree(1, 2, 3, 4);
    Tree tree2 = new Tree(2, 3, 4, 5);
    Tree tree3 = new Tree(3, 4, 5, 6);
    Tree tree4 = new Tree(4, 5, 6, 7);

    setLeafDestination(tree1, tree2, 1);
    setLeafDestination(tree2, tree3, 1);
    setLeafDestination(tree1, tree4, 0);

    Assert.assertTrue(tree3.getDestinationLeaf(0).isLinkedTo(tree1));
    Assert.assertTrue(tree3.getDestinationLeaf(1).isLinkedTo(tree1));
    Assert.assertTrue(tree3.getDestinationLeaf(0).isLinkedTo(tree2));
    Assert.assertTrue(tree3.getDestinationLeaf(1).isLinkedTo(tree2));
    Assert.assertTrue(tree3.getDestinationLeaf(0).isLinkedTo(tree3));
    Assert.assertTrue(tree3.getDestinationLeaf(1).isLinkedTo(tree3));
    Assert.assertFalse(tree3.getDestinationLeaf(0).isLinkedTo(tree4));
    Assert.assertFalse(tree3.getDestinationLeaf(1).isLinkedTo(tree4));

    Assert.assertTrue(tree4.getDestinationLeaf(0).isLinkedTo(tree1));
    Assert.assertTrue(tree4.getDestinationLeaf(1).isLinkedTo(tree1));
    Assert.assertFalse(tree4.getDestinationLeaf(0).isLinkedTo(tree2));
    Assert.assertFalse(tree4.getDestinationLeaf(1).isLinkedTo(tree2));
    Assert.assertFalse(tree4.getDestinationLeaf(0).isLinkedTo(tree3));
    Assert.assertFalse(tree4.getDestinationLeaf(1).isLinkedTo(tree3));

    Assert.assertTrue(tree2.getDestinationLeaf(0).isLinkedTo(tree2));
    Assert.assertTrue(tree2.getDestinationLeaf(0).isLinkedTo(tree1));
  }

  @Test
  public void testSameRoots() throws Exception {
    Tree tree1 = new Tree(1, 2, 3, 4);
    Tree tree2 = new Tree(2, 3, 4, 5);
    Tree tree3 = new Tree(3, 4, 5, 6);
    Tree tree4 = new Tree(4, 5, 6, 7);

    setLeafDestination(tree1, tree2, 1);
    setLeafDestination(tree2, tree3, 1);
    setLeafDestination(tree1, tree4, 0);

    Assert.assertTrue(tree3.getRootTree() == tree2.getRootTree()
        && tree2.getRootTree() == tree1.getRootTree()
        && tree3.getRootTree() == tree4.getRootTree());
  }

  private static void resetLeafDestination(Tree tree, int leaf){
    setLeafDestination(tree, null, leaf);
  }

  private static void resetAllLeafDestinations(Tree tree){
    tree.getDestinationLeaves().forEach(leaf -> leaf.setDestinationTree(null));
  }

  private static void setLeafDestination(Tree originalTree, Tree destinationTree, int leaf){
    originalTree.getDestinationLeaf(leaf).setDestinationTree(destinationTree);
  }
}