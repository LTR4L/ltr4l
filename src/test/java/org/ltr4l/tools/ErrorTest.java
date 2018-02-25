/*
 * Copyright 2018 org.LTR4L
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ltr4l.tools;

import org.junit.Assert;
import org.junit.Test;

public class ErrorTest {

  @Test
  public void testSquare() throws Exception {
    Error error = new Error.Square();

    Assert.assertEquals(0, error.error(1, 1), 0.001);
    Assert.assertEquals(0.5, error.error(2, 1), 0.001);
    Assert.assertEquals(0.5, error.error(1, 2), 0.001);
    Assert.assertEquals(2, error.error(3, 1), 0.001);
    Assert.assertEquals(2, error.error(1, 3), 0.001);

    Assert.assertEquals(0, error.der(1, 1), 0.001);
    Assert.assertEquals(1, error.der(2, 1), 0.001);
    Assert.assertEquals(-1, error.der(1, 2), 0.001);
    Assert.assertEquals(2, error.der(3, 1), 0.001);
    Assert.assertEquals(-2, error.der(1, 3), 0.001);
  }

  @Test
  public void testEntropy() throws Exception {
    Error error = new Error.Entropy();

    Assert.assertEquals(-2.302585, error.error(10, 1), 0.001);
    Assert.assertEquals(0, error.error(1, 1), 0.001);
    Assert.assertEquals(18.420681, error.error(0, 1), 0.001);

    Assert.assertEquals(-0.1, error.der(10, 1), 0.001);
    Assert.assertEquals(-1, error.der(1, 1), 0.001);
    Assert.assertEquals(-100000000, error.der(0, 1), 0.001);
  }

  @Test(expected = AssertionError.class)
  public void testEntropyOutputAssertError() throws Exception {
    Error error = new Error.Entropy();
    error.error(-Error.Entropy.DELTA, 1);
  }

  @Test(expected = AssertionError.class)
  public void testEntropyDerAssertError() throws Exception {
    Error error = new Error.Entropy();
    error.der(-Error.Entropy.DELTA,1);
  }
}
