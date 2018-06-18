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
