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

package org.ltr4l.nn;

import org.junit.Assert;
import org.junit.Test;

public class OptimizerTest {

  @Test
  public void testFactories() throws Exception {
    Assert.assertTrue(Optimizer.getFactory(Optimizer.Type.adam).getOptimizer() instanceof Optimizer.Adam);
    Assert.assertTrue(Optimizer.getFactory(Optimizer.Type.sgd).getOptimizer() instanceof Optimizer.SGD);
    Assert.assertTrue(Optimizer.getFactory(Optimizer.Type.momentum).getOptimizer() instanceof Optimizer.Momentum);
    Assert.assertTrue(Optimizer.getFactory(Optimizer.Type.nesterov).getOptimizer() instanceof Optimizer.Nesterov);

    Assert.assertTrue(Optimizer.getFactory(Optimizer.DEFAULT).getOptimizer() instanceof Optimizer.SGD);
  }

  @Test
  public void testOptimizeByAdam() throws Exception {
    Optimizer opt = Optimizer.getFactory(Optimizer.Type.adam).getOptimizer();

    Assert.assertEquals(-0.0199999, opt.optimize(0.1, 0.2, 1), 0.001);
    Assert.assertEquals(-0.0340523, opt.optimize(0.2, 0.2, 2), 0.001);
    Assert.assertEquals(-0.0473565, opt.optimize(0.3, 0.2, 3), 0.001);
    Assert.assertEquals(-0.0601400, opt.optimize(0.4, 0.2, 4), 0.001);
    Assert.assertEquals(-0.0724769, opt.optimize(0.5, 0.2, 5), 0.001);
  }

  @Test
  public void testOptimizeBySGD() throws Exception {
    Optimizer opt = Optimizer.getFactory(Optimizer.Type.sgd).getOptimizer();

    Assert.assertEquals(-0.02, opt.optimize(0.1, 0.2, 1), 0.001);
    Assert.assertEquals(-0.04, opt.optimize(0.2, 0.2, 2), 0.001);
    Assert.assertEquals(-0.06, opt.optimize(0.3, 0.2, 3), 0.001);
    Assert.assertEquals(-0.08, opt.optimize(0.4, 0.2, 4), 0.001);
    Assert.assertEquals(-0.1, opt.optimize(0.5, 0.2, 5), 0.001);
  }

  @Test
  public void testOptimizeByMomentum() throws Exception {
    Optimizer opt = Optimizer.getFactory(Optimizer.Type.momentum).getOptimizer();

    Assert.assertEquals(-0.0200000, opt.optimize(0.1, 0.2, 1), 0.001);
    Assert.assertEquals(-0.0590000, opt.optimize(0.2, 0.2, 2), 0.001);
    Assert.assertEquals(-0.1160500, opt.optimize(0.3, 0.2, 3), 0.001);
    Assert.assertEquals(-0.1902475, opt.optimize(0.4, 0.2, 4), 0.001);
    Assert.assertEquals(-0.2807351, opt.optimize(0.5, 0.2, 5), 0.001);
  }

  @Test
  public void testOptimizeByNesterov() throws Exception {
    Optimizer opt = Optimizer.getFactory(Optimizer.Type.nesterov).getOptimizer();

    Assert.assertEquals(-0.039000000, opt.optimize(0.1, 0.2, 1), 0.001);
    Assert.assertEquals(-0.096050000, opt.optimize(0.2, 0.2, 2), 0.001);
    Assert.assertEquals(-0.170247500, opt.optimize(0.3, 0.2, 3), 0.001);
    Assert.assertEquals(-0.260735125, opt.optimize(0.4, 0.2, 4), 0.001);
    Assert.assertEquals(-0.366698368, opt.optimize(0.5, 0.2, 5), 0.001);
  }
}
