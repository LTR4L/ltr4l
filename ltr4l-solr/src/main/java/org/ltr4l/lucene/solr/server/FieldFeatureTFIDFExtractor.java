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
