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

public class KernelParams {
  private double sigma;
  private double c; //linear bias
  private double d; //power

  public KernelParams(){
    sigma = 1d;
    c = 1d;
    d = 2d;
  }

  public void setC(double c) {
    this.c = c;
  }

  public void setD(double d) {
    this.d = d;
  }

  public void setSigma(double sigma) {
    this.sigma = sigma;
  }

  public double getC() {
    return c;
  }

  public double getD() {
    return d;
  }

  public double getSigma() {
    return sigma;
  }
}
