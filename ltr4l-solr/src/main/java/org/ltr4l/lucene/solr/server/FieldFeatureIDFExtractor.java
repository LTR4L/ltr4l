package org.ltr4l.lucene.solr.server;

import org.apache.lucene.search.Explanation;

import java.io.IOException;

public class FieldFeatureIDFExtractor implements FieldFeatureExtractor {

  private final int numDocs;
  private final int docFreq;
  private final float idf;

  public FieldFeatureIDFExtractor(int numDocs, int docFreq){
    assert numDocs >= docFreq;
    this.numDocs = numDocs + 1;
    this.docFreq = docFreq <= 0 ? 1 : docFreq;
    idf = (float)Math.log((double)this.numDocs/(double)this.docFreq);
  }

  @Override
  public float feature(int target) throws IOException {
    return idf;
  }

  @Override
  public Explanation explain(int target) throws IOException {
    return Explanation.match(idf, "log(numDocs: " + numDocs + "/docFreq: " + docFreq + ")");
  }
}
