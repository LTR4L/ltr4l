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

import java.util.*;
import java.util.stream.Collectors;

// TODO : Add options and methods for IDM
public class ClickModelAnalyzer {

  public ClickModelAnalyzer() { }

  public Map<String, Map<String, Float>> calcClickRate(List<ImpressionLog> impressionLogList) {
    Map<String, List<ImpressionLog>> queryGroup = impressionLogList.stream()
      .collect(Collectors.groupingBy(i -> i.getQuery(), Collectors.toList()));

    Map<String, Map<String, Float>> clickRates = new HashMap<>();
    Set<String> querySet = queryGroup.keySet();

    // TODO : Faster code
    for (String query : querySet) {
      Map<String, Integer> impressionCountMap = new HashMap<>();
      Map<String, Integer> clickCountMap = new HashMap<>();
      List<ImpressionLog> impressionLogs = queryGroup.get(query);
      for (ImpressionLog impressionLog : impressionLogs) {
        List<String> impressions = impressionLog.getImpressions();
        List<String> clicks = impressionLog.getClicks();
        count(impressions, impressionCountMap);
        count(clicks, clickCountMap);
      }
      Map<String, Float> clickRate = new HashMap<>();

      Set<String> urlSet = impressionCountMap.keySet();
      for (String url : urlSet) {
        Integer i = clickCountMap.get(url);
        clickRate.put(url, i == null ? 0.0f : i.floatValue() / impressionCountMap.get(url).floatValue());
      }
      clickRates.put(query, clickRate);
    }
    return clickRates;
  }

  private void count(List<String> strings, Map<String, Integer> countMap) {
    for (String string : strings) {
      Integer i = countMap.get(string);
      countMap.put(string, i == null ? 1 : i + 1);
    }
  }
}
