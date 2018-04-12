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
