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

import org.apache.solr.core.SolrResourceLoader;
import org.junit.Test;

import java.nio.file.Paths;

import static org.junit.Assert.*;

public class FeaturesConfigReaderTest {

  SolrResourceLoader loader = new SolrResourceLoader(Paths.get("src/test/resources/collection1/conf"));
  @Test
  public void testLoader() throws Exception {
    FeaturesConfigReader fcReader = new FeaturesConfigReader(loader, "ltr_features.conf");
    FeaturesConfigReader.FeatureDesc[] featureDescs = fcReader.getFeatureDescs();
    assertEquals(featureDescs.length, 4);
    assertEquals("name=TF in title,class=org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory,params=title",
            featureDescs[0].toString());
    assertEquals("name=TF in body,class=org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory,params=body",
            featureDescs[1].toString());
    assertEquals("name=IDF in title,class=org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory,params=title",
            featureDescs[2].toString());
    assertEquals("name=IDF in body,class=org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory,params=body",
            featureDescs[3].toString());
  }

  @Test
  public void testGetFeatureDesc() throws Exception {
    FeaturesConfigReader fcReader = new FeaturesConfigReader(loader,"ltr_features.conf");
    assertEquals("name=TF in title,class=org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory,params=title",
            fcReader.getFeatureDesc("TF in title").toString());
    assertEquals("name=TF in body,class=org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory,params=body",
            fcReader.getFeatureDesc("TF in body").toString());
    assertEquals("name=IDF in title,class=org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory,params=title",
            fcReader.getFeatureDesc("IDF in title").toString());
    assertEquals("name=IDF in body,class=org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory,params=body",
            fcReader.getFeatureDesc("IDF in body").toString());
  }

  @Test
  public void testLoadFactory() throws Exception {
    FeaturesConfigReader fcReader = new FeaturesConfigReader(loader,"ltr_features.conf");
    assertEquals(FieldFeatureTFExtractorFactory.class, FeaturesConfigReader.loadFactory(fcReader.getFeatureDesc("TF in title")).getClass());
    assertEquals(FieldFeatureTFExtractorFactory.class, FeaturesConfigReader.loadFactory(fcReader.getFeatureDesc("TF in body")).getClass());
    assertEquals(FieldFeatureIDFExtractorFactory.class, FeaturesConfigReader.loadFactory(fcReader.getFeatureDesc("IDF in title")).getClass());
    assertEquals(FieldFeatureIDFExtractorFactory.class, FeaturesConfigReader.loadFactory(fcReader.getFeatureDesc("IDF in body")).getClass());
  }
}
