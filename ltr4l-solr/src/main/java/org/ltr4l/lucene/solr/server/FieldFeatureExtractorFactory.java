package org.ltr4l.lucene.solr.server;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.index.*;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;

public abstract class FieldFeatureExtractorFactory {
  protected final String featureName;
  protected final String fieldName;
  protected IndexReader reader;
  protected Term[] terms;

  public FieldFeatureExtractorFactory(String featureName, String fieldName){
    this.featureName = featureName;
    this.fieldName = fieldName;
  }

  public String getFeatureName(){
    return featureName;
  }

  public String getFieldName(){
    return fieldName;
  }

  public void init(IndexReaderContext context, Term[] terms){
    assert context.isTopLevel;
    this.reader = context.reader();
    this.terms = terms;
  }

  @Override
  public String toString(){
    StringBuilder sb = new StringBuilder();
    sb.append(this.getClass().getName());
    sb.append('[');
    if(terms != null && terms.length > 0){
      for(int i = 0; i < terms.length; i++){
        if(i > 0) sb.append(',');
        sb.append(terms[i].toString());
      }
    }
    sb.append(']');
    return sb.toString();
  }

  @Override
  public boolean equals(Object o) {
    if(o == null || !(o instanceof FieldFeatureExtractorFactory)) return false;
    FieldFeatureExtractorFactory other = (FieldFeatureExtractorFactory)o;
    if(!this.getClass().equals(other.getClass())) return false;
    if(this.reader != other.reader) return false;
    if(this.terms == null){
      return other.terms == null;
    }
    else{
      if(this.terms.length != other.terms.length) return false;
      for(int i = 0; i < this.terms.length; i++){
        if(!this.terms[i].equals(other.terms[i])) return false;
      }
    }

    return true;
  }

  @Override
  public int hashCode() {
    final int prime = 53;
    int result = 1;
    result = prime * result + getClass().hashCode();
    result = prime * result + reader.hashCode();
    if(terms == null) return result;

    for(int i = 0; i < terms.length; i++){
      result = prime * result + terms[i].hashCode();
    }
    return result;
  }

  public Term[] getTerms() { return terms; }

  public abstract FieldFeatureExtractor[] create(LeafReaderContext context, Set<Integer> allDocs) throws IOException;

  protected TermsEnum getTermsEnum(LeafReaderContext context, Term term) throws IOException {
    Terms terms = context.reader().terms(term.field());
    if (terms == null) {
      return null;
    }
    final TermsEnum termsEnum = terms.iterator();
    if (termsEnum.seekExact(term.bytes())) {
      return termsEnum;
    } else {
      return null;
    }
  }

  public static Term[] terms(String fieldName, String qstr, Analyzer analyzer){
    List<Term> terms = new ArrayList<Term>();
    TokenStream stream = analyzer.tokenStream(fieldName, qstr);
    CharTermAttribute termAtt = stream.addAttribute(CharTermAttribute.class);
    try {
      stream.reset();
      while(stream.incrementToken()){
        terms.add(new Term(fieldName, termAtt.toString()));
      }
      stream.end();
      stream.close();
    } catch (IOException e) {
      throw new RuntimeException(e);
    }

    return terms.toArray(new Term[terms.size()]);
  }
}
