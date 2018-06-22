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

package org.ltr4l.click;

public class ClickRateClassifier {

  protected final Double[] borders;

  public ClickRateClassifier(String borderListStr) throws ClassCastException {
    String[] bordersStr = borderListStr.replace(" ", "").split(",");
    borders = new Double[borderListStr.length()];

    for (int i = 0; i < bordersStr.length; i++) {
      borders[i] = Double.valueOf(bordersStr[i]);
    }
    sort(borders);
  }

  public int classify(double value) {
    if (value <= borders[0]) {
      return 0;
    }
    for (int i = 1; i < borders.length; i++) {
      if (value <= borders[i] && value > borders[i - 1]) {
        return i;
      }
    }
    return borders.length;
  }

  public int classify(float value) {
    return classify((double)value);
  }

  public int classify(int value) {
    return classify((double)value);
  }

  public int classify(long value) {
    return classify((double)value);
  }

  private void sort(Double[] borders) {
    //TODO: write sort
  }
}
