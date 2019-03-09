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
package org.ltr4l.conversion;

import com.fasterxml.jackson.annotation.JsonProperty;

import java.util.List;
import java.util.Map;


public class SolrLTRModel {
  @JsonProperty("class")
  public String clazz;
  @JsonProperty("name")
  public String name;
  @JsonProperty("features")
  public List<Feature> features;
  @JsonProperty("params")
  public Map<String, Object> params;

  public static class Feature {
    public String name;

    public Feature(String name) {
      this.name = name;
    }
  }
}
