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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

public class VectorMath {

  private static void checkVectors(List<Double> A, List<Double> B) {
    Objects.requireNonNull(A);
    Objects.requireNonNull(B);
    assert(A.size() == B.size());
  }

  public static double dot(List<Double> A, List<Double> B) {
    checkVectors(A, B);
    double dotProd = 0;
    for(int i = 0; i < A.size(); i++)
      dotProd += A.get(i) * B.get(i);
    return dotProd;
  }

  public static double norm2(List<Double> A) {
    //Squared norm
    double norm = 0d;
    for(double elem : A)
      norm += elem * elem;
    return norm;
  }

  public static double norm(List<Double> A) {
    return Math.sqrt(norm2(A));
  }

  public static List<Double> diff(List<Double> A, List<Double> B){
    checkVectors(A, B);
    List<Double> diff = new ArrayList<>();
    for(int i = 0; i < A.size(); i++)
      diff.add(A.get(i) - B.get(i));
    return diff;
  }


}
