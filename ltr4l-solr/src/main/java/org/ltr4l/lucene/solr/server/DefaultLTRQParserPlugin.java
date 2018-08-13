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

public class DefaultLTRQParserPlugin extends QParserPlugin {
  List<FieldFeatureExtractorFactory> featuresSpec = new ArrayList<FieldFeatureExtractorFactory>();
  Ranker ranker;

  @Override
  public void init(NamedList args){
    NamedList settings = (NamedList)args.get("settings");
    String featuresFileName = (String)settings.get("features");
    String modelFileName = (String)settings.get("model");

    FeaturesConfigReader fcReader;
    DefaultLTRModelReader dlmReader;
    try {
      fcReader = new FeaturesConfigReader(featuresFileName);
      dlmReader = new DefaultLTRModelReader(modelFileName);

      FeaturesConfigReader.FeatureDesc[] featureDescs = fcReader.getFeatureDescs();
      for (FeaturesConfigReader.FeatureDesc featureDesc : featureDescs) {
        if(featureDesc == null){
          continue;
        }
        FieldFeatureExtractorFactory dfeFactory = FeaturesConfigReader.loadFactory(featureDesc);
        featuresSpec.add(dfeFactory);
      }
      ranker = dlmReader.getRanker();
    } catch (IOException ioe) {
      ioe.printStackTrace();
    }
  }

  @Override
  public QParser createParser(String query, SolrParams localParams, SolrParams params, SolrQueryRequest req) {
    return new DefaultLTRQParser(query, localParams, params, req, ranker);
  }

  public class DefaultLTRQParser extends QParser {
    Ranker ranker;
    public DefaultLTRQParser(String query, SolrParams localParams, SolrParams params, SolrQueryRequest req, Ranker ranker) {
      super(query, localParams, params, req);
      this.ranker = ranker;
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

      return new DefaultLTRQuery(featuresSpec, ranker);
    }
  }

}
