package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;

import java.io.IOException;
import java.util.Set;

public class FieldFeatureIDFExtractorFactory extends FieldFeatureExtractorFactory {
  private int numDocs;

  public FieldFeatureIDFExtractorFactory(String featureName, String fieldName){
    super(featureName, fieldName);
  }

  @Override
  public void init(IndexReaderContext context, Term[] terms) {
    super.init(context, terms);
    numDocs = reader.numDocs();
  }

  @Override
  public FieldFeatureExtractor[] create(LeafReaderContext context, Set<Integer> allDocs) throws IOException {
    FieldFeatureExtractor[] extractors = new FieldFeatureExtractor[terms.length];
    int i = 0;
    for(Term term: terms){
      extractors[i++] = new FieldFeatureIDFExtractor(numDocs, reader.docFreq(term));
    }
    return extractors;
  }
}
