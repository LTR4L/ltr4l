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
import org.apache.lucene.index.Term;
import org.apache.lucene.util.LuceneTestCase;

import java.util.ArrayList;
import java.util.List;

public abstract class AbstractLTRQueryTestCase extends LuceneTestCase {

  protected FieldFeatureExtractorFactory getTF(String featureName, String fieldName){
    FieldFeatureExtractorFactory factory = new FieldFeatureTFExtractorFactory(featureName, fieldName);
    return factory;
  }

  protected FieldFeatureExtractorFactory getTF(String featureName, String fieldName, IndexReaderContext context, Term... terms){
    FieldFeatureExtractorFactory factory = new FieldFeatureTFExtractorFactory(featureName, fieldName);
    if(context != null){
      factory.init(context, terms);
    }
    return factory;
  }

  protected FieldFeatureExtractorFactory getIDF(String featureName, String fieldName){
    FieldFeatureExtractorFactory factory = new FieldFeatureIDFExtractorFactory(featureName, fieldName);
    return factory;
  }

  protected FieldFeatureExtractorFactory getIDF(String featureName, String fieldName, IndexReaderContext context, Term... terms){
    FieldFeatureExtractorFactory factory = new FieldFeatureIDFExtractorFactory(featureName, fieldName);
    if(context != null){
      factory.init(context, terms);
    }
    return factory;
  }

  protected FieldFeatureExtractorFactory getTFIDF(String featureName, String fieldName){
    FieldFeatureExtractorFactory factory = new FieldFeatureTFIDFExtractorFactory(featureName, fieldName);
    return factory;
  }

  protected FieldFeatureExtractorFactory getTFIDF(String featureName, String fieldName, IndexReaderContext context, Term... terms){
    FieldFeatureExtractorFactory factory = new FieldFeatureTFIDFExtractorFactory(featureName, fieldName);
    if(context != null){
      factory.init(context, terms);
    }
    return factory;
  }

  protected FieldFeatureExtractorFactory getSV(String featureName, String fieldName){
    FieldFeatureExtractorFactory factory = new FieldFeatureStoredValueExtractorFactory(featureName, fieldName);
    return factory;
  }

  protected FieldFeatureExtractorFactory getSV(String featureName, String fieldName, IndexReaderContext context, Term... terms){
    FieldFeatureExtractorFactory factory = new FieldFeatureStoredValueExtractorFactory(featureName, fieldName);
    if(context != null){
      factory.init(context, terms);
    }
    return factory;
  }

  protected List<FieldFeatureExtractorFactory> buildFeaturesSpec(FieldFeatureExtractorFactory... factories){
    List<FieldFeatureExtractorFactory> featuresSpec = new ArrayList<FieldFeatureExtractorFactory>();
    for(FieldFeatureExtractorFactory factory: factories){
      featuresSpec.add(factory);
    }
    return featuresSpec;
  }
}
