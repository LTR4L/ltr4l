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
import org.ltr4l.query.Document;
import org.ltr4l.tools.Config;
import org.ltr4l.tools.Error;

import java.util.List;
import java.util.Map;

public abstract class AbstractSVM<C extends AbstractSVM.SVMConfig> extends Ranker<C> {
  protected final Kernel kernel;

  protected AbstractSVM(Kernel kernel){
    this.kernel = kernel;
  }

  public abstract void optimize();

  public static class SVMConfig extends Config {
    @JsonIgnore
    public String getSVMWeightInit(){
      return getString(params, "weightInit", SVMInitializer.Type.UNIFORM.name());
    }
    @JsonIgnore
    public double getLearningRate() { return getReqDouble(params, "learningRate"); }
    @JsonIgnore
    public Solver.Type getOptimizer() { return Solver.Type.get(getString(params, "optimizer", "sgd")); }
    @JsonIgnore
    public boolean getMetricOption() {return getBoolean(params, "optMetric", false);}
    @JsonIgnore
    public static boolean getBoolean(Map<String, Object> params, String name, boolean defValue){
      Object obj = params.get(name);
      if(obj == null){
        return defValue;
      }
      return Boolean.parseBoolean(obj.toString());
    }
    @JsonIgnore
    public Kernel getKernel() {return Kernel.getKernel(getReqString(params, "kernel"));}
    @JsonIgnore
    public boolean dataIsSVMFormat() {
      return getBoolean(params, "isSVMData", false);
    }
  }
}
