package org.ltr4l.svm;

import org.junit.Assert;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

public class KernelTest {
  private KernelParams defaultParams;
  private List<Double> A;
  private List<Double> B;

  @Before
  public void setUp() throws Exception {
    defaultParams = new KernelParams();
    A = new ArrayList<>();
    for(double num : new double[] {1d, 2d, 3d, 4d})
      A.add(num);
    B = new ArrayList<>();
    for(double num : new double[] {5d, 6d, 7d, 8d})
      B.add(num);
  }

  @Test
  public void testIdentity() throws Exception {
    Kernel kernel = Kernel.Type.IDENTITY;
    Assert.assertEquals(70d, kernel.similarityK(A, B), 0.01);
  }

  @Test
  public void testLinear() throws Exception {
    Kernel kernel = Kernel.Type.LINEAR;
    Assert.assertEquals(71d, kernel.similarityK(A, B, defaultParams), 0.01);
  }

  @Test
  public void testPolynomial() throws Exception {
    Kernel kernel = Kernel.Type.POLYNOMIAL;
    KernelParams params = new KernelParams().setSigma(2).setC(2);
    Assert.assertEquals(142d * 142d, kernel.similarityK(A, B, params), 0.01);
    Assert.assertEquals(71d * 71d, kernel.similarityK(A, B), 0.01);
  }

  @Test
  public void testGaussian() throws Exception {
    Kernel kernel = Kernel.Type.GAUSSIAN;
    KernelParams params = new KernelParams().setSigma(2);
    Assert.assertEquals(Math.exp(-64/8d), kernel.similarityK(A, B, params), 0.000001);
    Assert.assertEquals(Math.exp(-64/2d), kernel.similarityK(A, B), 0.000001);
  }

  @Test
  public void testExponential() throws Exception {
    Kernel kernel = Kernel.Type.EXPONENTIAL;
    KernelParams params = new KernelParams().setSigma(2);
    Assert.assertEquals(Math.exp(-8/8d), kernel.similarityK(A, B, params), 0.000001);
    Assert.assertEquals(Math.exp(-8/2d), kernel.similarityK(A, B), 0.000001);
  }

  @Test
  public void testLaplacian() throws Exception {
    Kernel kernel = Kernel.Type.LAPLACIAN;
    KernelParams params = new KernelParams().setSigma(2);
    Assert.assertEquals(Math.exp(-8/2d), kernel.similarityK(A, B, params), 0.000001);
    Assert.assertEquals(Math.exp(-8), kernel.similarityK(A, B), 0.000001);
  }

  @Test
  public void testDefault() throws Exception {
    assertDefault(Kernel.Type.IDENTITY, 0.01);
    assertDefault(Kernel.Type.LINEAR, 0.01);
    assertDefault(Kernel.Type.POLYNOMIAL, 0.01);
    assertDefault(Kernel.Type.GAUSSIAN, 0.000001);
    assertDefault(Kernel.Type.EXPONENTIAL, 0.000001);
    assertDefault(Kernel.Type.LAPLACIAN, 0.000001);
  }

  private void assertDefault(Kernel kernel, double delta) throws Exception {
    Assert.assertEquals(kernel.similarityK(A, B, defaultParams), kernel.similarityK(A, B), delta);
  }


}