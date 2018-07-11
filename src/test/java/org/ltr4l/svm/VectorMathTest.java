package org.ltr4l.svm;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class VectorMathTest {
  private List<Double> A;
  private List<Double> B;

  @Before
  public void setUp() throws Exception {
    A = new ArrayList<>();
    for(double num : new double[] {1d, 2d, 3d, 4d})
      A.add(num);
    B = new ArrayList<>();
    for(double num : new double[] {5d, 7d, 9d, 11d})
      B.add(num);
  }

  @Test
  public void dot() throws Exception {
    Assert.assertEquals(100d, VectorMath.dot(A, B), 0.01);
  }

  @Test
  public void norm2() throws Exception {
    Assert.assertEquals(30d, VectorMath.norm2(A), 0.01);
    Assert.assertEquals(276d, VectorMath.norm2(B), 0.01);
  }

  @Test
  public void norm() throws Exception {
    Assert.assertEquals(Math.sqrt(30d), VectorMath.norm(A), 0.01);
    Assert.assertEquals(Math.sqrt(276d), VectorMath.norm(B), 0.01);
  }

  @Test
  public void diff() throws Exception {
    double diff = 4d;
    for(double elem : VectorMath.diff(A, B)){
      Assert.assertEquals(-diff, elem, 0.01);
      diff++;
    }
  }
}