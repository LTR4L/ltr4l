package org.ltr4l.lucene.solr.server;

import org.apache.lucene.index.LeafReaderContext;

import java.io.IOException;
import java.util.Set;

public class FieldFeatureStoredValueExtractorFactory extends FieldFeatureExtractorFactory {

  public FieldFeatureStoredValueExtractorFactory(String featureName, String fieldName){
    super(featureName, fieldName);
  }

  @Override
  public FieldFeatureExtractor[] create(LeafReaderContext context, Set<Integer> allDocs) throws IOException {
    FieldFeatureExtractor[] extractors = new FieldFeatureExtractor[1];
    extractors[0] = new FieldFeatureStoredValueExtractor(context.reader(), fieldName);
    return extractors;
  }
}
