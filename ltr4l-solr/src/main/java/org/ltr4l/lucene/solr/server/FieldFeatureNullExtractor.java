package org.ltr4l.lucene.solr.server;

import org.apache.lucene.search.Explanation;

import java.io.IOException;

public class FieldFeatureNullExtractor implements FieldFeatureExtractor {
  @Override
  public float feature(int doc) throws IOException {
    return 0;
  }

  @Override
  public Explanation explain(int doc) throws IOException {
    return Explanation.noMatch("no matching terms");
  }
}
