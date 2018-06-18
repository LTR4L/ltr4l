package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.search.Explanation;

import java.io.IOException;

public class FieldFeatureTFIDFExtractor implements FieldFeatureExtractor {

  private final PostingsEnum pe;
  private final int numDocs;
  private final int docFreq;
  private final float idf;

  public FieldFeatureTFIDFExtractor(PostingsEnum pe, int numDocs, int docFreq){
    this.pe = pe;
    assert numDocs >= docFreq;
    this.numDocs = numDocs + 1;
    this.docFreq = docFreq <= 0 ? 1 : docFreq;
    idf = (float)Math.log((double)this.numDocs/(double)this.docFreq);
  }

  @Override
  public float feature(int target) throws IOException {
    int current = pe.docID();
    if(current < target){
      current = pe.advance(target);
    }
    if(current == target){
      return pe.freq() * idf;
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
      float score = freq * idf;
      Explanation eTf = Explanation.match(freq, "TF = freq = " + freq);
      Explanation eIdf = Explanation.match(idf, "IDF = log(" + numDocs + "/" + docFreq + ")");
      return Explanation.match(score, "TF * IDF:", eTf, eIdf);
    }
    else{
      return Explanation.noMatch("no matching terms");
    }
  }
}
