/*
 * Copyright 2018 org.LTR4L
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.ltr4l.lucene.solr.server;

import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Query;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Set;

public abstract class AbstractLTRQuery extends Query {

  protected final List<FieldFeatureExtractorFactory> featuresSpec;

  public AbstractLTRQuery(List<FieldFeatureExtractorFactory> featuresSpec){
    this.featuresSpec = featuresSpec;
  }

  @Override
  public String toString(String ignored) {
    StringBuilder sb = new StringBuilder();
    sb.append(this.getClass().getName());
    sb.append(" featuresSpec=[");
    if(featuresSpec != null){
      for(int i = 0; i < featuresSpec.size(); i++){
        if(i > 0) sb.append(',');
        sb.append(featuresSpec.get(i).toString());
      }
    }
    sb.append(']');
    return sb.toString();
  }

  @Override
  public boolean equals(Object o) {
    if(o == null || !(o instanceof AbstractLTRQuery)) return false;
    AbstractLTRQuery other = (AbstractLTRQuery)o;
    return equalsFeaturesSpec(other.featuresSpec);
  }

  public int hashCode(int prime) {
    int result = 1;
    if(featuresSpec != null){
      for(FieldFeatureExtractorFactory factory: featuresSpec){
        result = prime * result + factory.hashCode();
      }
    }

    return result;
  }

  protected boolean equalsFeaturesSpec(List<FieldFeatureExtractorFactory> oSpec){
    if(this.featuresSpec == null){
      return oSpec == null;
    }
    else{
      if(oSpec == null) return false;
      else{
        if(this.featuresSpec.size() != oSpec.size()) return false;
        else{
          for(int i = 0; i < this.featuresSpec.size(); i++){
            if(!this.featuresSpec.get(i).equals(oSpec.get(i))) return false;
          }
          return true;
        }
      }
    }
  }

  protected DocIdSetIterator getIterator(Set<Integer> allDocs){
    final List<Integer> docs = new ArrayList<Integer>(allDocs);
    Collections.sort(docs);
    return new DocIdSetIterator() {
      int pos = -1;
      int docId = -1;

      @Override
      public int docID() {
        return docId;
      }

      @Override
      public int nextDoc() throws IOException {
        pos++;
        docId = pos >= docs.size() ? NO_MORE_DOCS : docs.get(pos);
        return docId;
      }

      @Override
      public int advance(int target) throws IOException {
        while(docId < target){
          nextDoc();
        }
        return docId;
      }

      @Override
      public long cost() {
        // TODO: set proper cost value...
        return 1;
      }
    };
  }
}
