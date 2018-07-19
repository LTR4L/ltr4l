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
package org.ltr4l.svm;

import com.fasterxml.jackson.annotation.JsonIgnore;
import org.ltr4l.Ranker;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;

public abstract class AbstractSVM<C extends AbstractSVM.SVMConfig> extends Ranker<C> {
  protected final Kernel kernel;
  protected final KernelParams params;

  protected AbstractSVM(Kernel kernel){
    this.kernel = kernel;
    this.params = new KernelParams();
  }

  public abstract void optimize(SVMOptimizer optimizer, Error error, double output, double target);

  public KernelParams getParams() {
    return params;
  }

  public static class SVMConfig extends Config {
    @JsonIgnore
    public String getSVMWeightInit(){
      return getString(params, "svmInit", SVMInitializer.Type.UNIFORM.name());
    }
    @JsonIgnore
    public double getLearningRate() { return getReqDouble(params, "learningRate"); }
    @JsonIgnore
    public SVMOptimizer getOptimizer() { return SVMOptimizer.Factory.get(getString(params, "optimizer", "sgd")); }
  }
}
