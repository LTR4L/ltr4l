package org.ltr4l.query;

import java.util.ArrayList;
import java.util.List;

public class Document {
  private int Label;
  private List<Double> features;

  Document() {
    features = new ArrayList<>();
  }

  public int getLabel() {
    return Label;
  }

  protected void setLabel(int newLabel) {
    Label = newLabel;
  }

  public List<Double> getFeatures() {
    return features;
  }

  protected void addFeature(double feature) {
    features.add(feature);
  }
}
