package org.ltr4l.click;

import org.junit.Assert;
import org.junit.Test;

public class ClickRateClassifierTest {
  private final String borderList1 = "1.0, 3.0, 5.0, 7.0";
  private final String borderList2 = "1.0";

  @Test
  public void testClassify() throws Exception{
    ClickRateClassifier classifier1 = new ClickRateClassifier(borderList1);

    Assert.assertEquals(0, classifier1.classify(1.0d));
    Assert.assertEquals(0, classifier1.classify(0.1f));
    Assert.assertEquals(1, classifier1.classify(2));
    Assert.assertEquals(1, classifier1.classify(2L));
    Assert.assertEquals(2, classifier1.classify(4.3));
    Assert.assertEquals(4, classifier1.classify(7.01));

    ClickRateClassifier classifier2 = new ClickRateClassifier(borderList2);
    Assert.assertEquals(0, classifier1.classify(1));
    Assert.assertEquals(1, classifier1.classify(1.1));
  }
}
