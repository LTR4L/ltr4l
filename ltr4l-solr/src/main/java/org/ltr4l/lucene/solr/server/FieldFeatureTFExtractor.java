package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.search.Explanation;

import java.io.IOException;

public class FieldFeatureTFExtractor implements FieldFeatureExtractor {

  private final PostingsEnum pe;

  public FieldFeatureTFExtractor(PostingsEnum pe){
    this.pe = pe;
  }

  @Override
  public float feature(int target) throws IOException {
    int current = pe.docID();
    if(current < target){
      current = pe.advance(target);
    }
    if(current == target){
      return pe.freq();
    }
    else{
      return 0;
    }
  }

  @Override
  public Explanation explain(int target) throws IOException {
    int current = pe.docID();
    if(current < target){
      current = pe.advance(target);
    }
    if(current == target){
      int freq = pe.freq();
      return Explanation.match(freq, "freq: " + freq);
    }
    else{
      return Explanation.noMatch("no matching terms");
    }
  }
}
