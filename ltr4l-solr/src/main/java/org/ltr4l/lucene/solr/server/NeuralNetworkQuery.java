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

import org.apache.commons.pool2.impl.GenericObjectPool;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import org.ltr4l.Ranker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class NeuralNetworkQuery extends AbstractLTRQuery {
  GenericObjectPool<Ranker> rankerPool;
//  private Ranker ranker;
//  public NeuralNetworkQuery(List<FieldFeatureExtractorFactory> featuresSpec, Ranker ranker){
  public NeuralNetworkQuery(List<FieldFeatureExtractorFactory> featuresSpec, GenericObjectPool<Ranker> rankerPool){
    super(featuresSpec);
    this.rankerPool = rankerPool;
  }

  @Override
  public int hashCode() {
    final int prime = 71;
    int result = super.hashCode(prime);
    return result;
  }


  @Override
  public Weight createWeight(IndexSearcher searcher, boolean needsScores, float boost) throws IOException {
    return new NeuralNetworkQuery.NeuralNetworkWeight(this);
  }

  public final class NeuralNetworkWeight extends Weight {

    protected NeuralNetworkWeight(Query query){
      super(query);
    }

    /**
     * Expert: adds all terms occurring in this query to the terms set. If the
     * {@link Weight} was created with {@code needsScores == true} then this
     * method will only extract terms which are used for scoring, otherwise it
     * will extract all terms which are used for matching.
     */
    @Override
    public void extractTerms(Set<Term> set) {
      for(FieldFeatureExtractorFactory factory: featuresSpec){
        // TODO: need to remove redundant terms...
        for(Term term: factory.getTerms()){
          set.add(term);
        }
      }
    }

    @Override
    public Explanation explain(LeafReaderContext leafReaderContext, int doc) throws IOException {
      NeuralNetworkDefaultScorer scorer = (NeuralNetworkDefaultScorer)scorer(leafReaderContext);
      if(scorer != null){
        int newDoc = scorer.iterator().advance(doc);
        if (newDoc == doc) {
          StringBuilder sb = new StringBuilder();
          return Explanation.match(scorer.score(),
            String.format("is the index of %s > %f sum of:", sb.toString()),
            scorer.subExplanations(doc));
        }
      }
      return Explanation.noMatch("no matching terms");
    }

    @Override
    public Scorer scorer(LeafReaderContext context) throws IOException {
      List<FieldFeatureExtractor[]> spec = new ArrayList<FieldFeatureExtractor[]>();
      Set<Integer> allDocs = new HashSet<Integer>();
      for(FieldFeatureExtractorFactory factory: featuresSpec){
        FieldFeatureExtractor[] extractors = factory.create(context, allDocs);
        spec.add(extractors);
      }

      if(allDocs.size() == 0) {
        return null;
      } else {
//        return new NeuralNetworkDefaultScorer(this, spec, getIterator(allDocs), ranker);
        return new NeuralNetworkDefaultScorer(this, spec, getIterator(allDocs), rankerPool);
      }
    }

    @Override
    public boolean isCacheable(LeafReaderContext leafReaderContext) {
      return false;
    }
  }
}
