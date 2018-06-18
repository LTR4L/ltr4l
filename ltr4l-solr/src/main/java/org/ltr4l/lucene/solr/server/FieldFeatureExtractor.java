package org.ltr4l.lucene.solr.server;

import org.apache.lucene.search.Explanation;

import java.io.IOException;

public interface FieldFeatureExtractor {
  public float feature(int doc) throws IOException;
  public Explanation explain(int doc) throws IOException;
}
