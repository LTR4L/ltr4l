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
import org.apache.commons.pool2.impl.GenericObjectPoolConfig;
import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.search.Query;
import org.apache.solr.common.params.SolrParams;
import org.apache.solr.common.util.NamedList;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.schema.FieldType;
import org.apache.solr.search.QParser;
import org.apache.solr.search.QParserPlugin;
import org.apache.solr.search.SyntaxError;
import org.ltr4l.Ranker;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class NeuralNetworkQParserPlugin extends QParserPlugin {
  List<FieldFeatureExtractorFactory> featuresSpec = new ArrayList<FieldFeatureExtractorFactory>();
  GenericObjectPool<Ranker> rankerPool;

  @Override
  public void init(NamedList args){
    NamedList settings = (NamedList)args.get("settings");
    String featuresFileName = (String)settings.get("features");
    String modelFileName = (String)settings.get("model");
    int poolSize = Integer.valueOf((String)settings.get("poolSize"));

    FeaturesConfigReader fcReader = null;
    NeuralNetworkModelReader nnmReader = null;

    try {
      fcReader = new FeaturesConfigReader(featuresFileName);
      nnmReader = new NeuralNetworkModelReader(modelFileName);

      FeaturesConfigReader.FeatureDesc[] featureDescs = fcReader.getFeatureDescs();
      for (FeaturesConfigReader.FeatureDesc featureDesc : featureDescs) {
        if(featureDesc == null){
          System.err.println("feature : null");
          continue;
        }
        System.err.println("feature : " + featureDesc.name);
        FieldFeatureExtractorFactory dfeFactory = FeaturesConfigReader.loadFactory(featureDesc);
        featuresSpec.add(dfeFactory);
      }
      createRankerPool(poolSize, nnmReader);
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }
  }

  @Override
  public QParser createParser(String query, SolrParams localParams, SolrParams params, SolrQueryRequest req) {
/*
    Ranker ranker;
    try {
      ranker = rankerPool.borrowObject();
    } catch (Exception e) {
      e.printStackTrace();
      return null;
    }
*/
//    return new NeuralNetworkQParser(query, localParams, params, req, ranker);
    return new NeuralNetworkQParser(query, localParams, params, req, rankerPool);

  }

  private void createRankerPool(int poolSize, NeuralNetworkModelReader nnmReader) {
    GenericObjectPoolConfig genericObjectPoolConfig = new GenericObjectPoolConfig();
    genericObjectPoolConfig.setMaxTotal(poolSize);
    genericObjectPoolConfig.setBlockWhenExhausted(false);

    rankerPool = new GenericObjectPool<>(new PooledRankerFactory(nnmReader), genericObjectPoolConfig);

    //TODO: Smarter code.
    List<Ranker> rankerList = new ArrayList<>(poolSize);
    try {
      for (int i = 0; i< poolSize; i++) {
        rankerList.add(rankerPool.borrowObject());
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
    for (Ranker ranker : rankerList) {
      rankerPool.returnObject(ranker);
    }
  }


  public class NeuralNetworkQParser extends QParser {
//    Ranker ranker;
//    public NeuralNetworkQParser(String query, SolrParams localParams, SolrParams params, SolrQueryRequest req, Ranker ranker) {
    GenericObjectPool<Ranker> rankerPool;
    public NeuralNetworkQParser(String query, SolrParams localParams, SolrParams params, SolrQueryRequest req, GenericObjectPool<Ranker> rankerPool) {
      super(query, localParams, params, req);
      this.rankerPool = rankerPool;
    }

    @Override
    public Query parse() throws SyntaxError {
      IndexReaderContext context = req.getSearcher().getTopReaderContext();
      for(FieldFeatureExtractorFactory factory: featuresSpec){
        String fieldName = factory.getFieldName();
        FieldType fieldType = req.getSchema().getFieldType(fieldName);
        Analyzer analyzer = fieldType.getQueryAnalyzer();
        factory.init(context, FieldFeatureExtractorFactory.terms(fieldName, qstr, analyzer));
      }

//      return new NeuralNetworkQuery(featuresSpec, ranker);
      return new NeuralNetworkQuery(featuresSpec, rankerPool);
    }
  }

}
