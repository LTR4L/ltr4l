package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.*;

import java.io.IOException;
import java.util.Set;

public class FieldFeatureTFIDFExtractorFactory extends FieldFeatureExtractorFactory {
  private int numDocs;

  public FieldFeatureTFIDFExtractorFactory(String featureName, String fieldName){
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
      final TermsEnum termsEnum = getTermsEnum(context, term);
      if (termsEnum == null) {
        extractors[i] = new FieldFeatureNullExtractor();
      }
      else{
        extractors[i] = new FieldFeatureTFIDFExtractor(termsEnum.postings(null, PostingsEnum.FREQS), numDocs, reader.docFreq(term));
        // get it twice without reuse to clone it...
        PostingsEnum docs = termsEnum.postings(null, PostingsEnum.FREQS);
        for(int docId = docs.nextDoc(); docId != PostingsEnum.NO_MORE_DOCS; docId = docs.nextDoc()){
          allDocs.add(docId);
        }
      }
      i++;
    }
    return extractors;
  }
}
