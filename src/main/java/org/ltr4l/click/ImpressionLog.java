package org.ltr4l.click;

import java.util.List;

public class ImpressionLog {
  private String query;
  private List<String> impressions;
  private List<String> clicks;

  public ImpressionLog(String query, List<String> impressions, List<String> clicks) {
    this.query = query;
    this.impressions = impressions;
    this.clicks = clicks;
  }

  public String getQuery() {
    return query;
  }

  public List<String> getImpressions() {
    return impressions;
  }

  public List<String> getClicks() {
    return clicks;
  }
}