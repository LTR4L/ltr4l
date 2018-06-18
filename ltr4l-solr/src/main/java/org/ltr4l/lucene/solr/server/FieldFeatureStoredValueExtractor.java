package org.ltr4l.lucene.solr.server;

import org.apache.lucene.document.Document;
import org.apache.lucene.index.LeafReader;
import org.apache.lucene.search.Explanation;

import java.io.IOException;

public class FieldFeatureStoredValueExtractor implements FieldFeatureExtractor {

  private final LeafReader reader;
  private final String fieldName;

  public FieldFeatureStoredValueExtractor(LeafReader reader, String fieldName){
    this.reader = reader;
    this.fieldName = fieldName;
  }

  @Override
  public float feature(int target) throws IOException {
    return getFloatValue(target);
  }

  @Override
  public Explanation explain(int target) throws IOException {
    return Explanation.match(feature(target), "value(" + getStrValue(target) + ")");
  }

  float getFloatValue(int target) throws IOException {
    String value = getStrValue(target);
    if(value != null){
      try{
        return Float.parseFloat(value);
      }
      catch (NumberFormatException e){
        return 0;
      }
    }
    return 0;
  }

  String getStrValue(int target) throws IOException {
    Document doc = reader.document(target);
    return doc == null ? null : doc.get(fieldName);
  }
}
