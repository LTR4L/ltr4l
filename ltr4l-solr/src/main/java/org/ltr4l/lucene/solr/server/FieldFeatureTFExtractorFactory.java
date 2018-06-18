package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.PostingsEnum;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.TermsEnum;

import java.io.IOException;
import java.util.Set;

public class FieldFeatureTFExtractorFactory extends FieldFeatureExtractorFactory {

  public FieldFeatureTFExtractorFactory(String featureName, String fieldName){
    super(featureName, fieldName);
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
        extractors[i] = new FieldFeatureTFExtractor(termsEnum.postings(null, PostingsEnum.FREQS));
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
