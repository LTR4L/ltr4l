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
import org.ltr4l.nn.RankNetMLP;

import java.nio.file.Paths;

import static org.junit.Assert.*;

public class DefaultLTRModelReaderTest {
  SolrResourceLoader loader = new SolrResourceLoader(Paths.get("src/test/resources/collection1/conf"));

  @Test
  public void testLoader() throws Exception {
    DefaultLTRModelReader dlmReader = new DefaultLTRModelReader(loader,"ranknet_model.conf");
    assertEquals("RankNet".toLowerCase(), dlmReader.getAlgorithm().toLowerCase());
    assertTrue(dlmReader.getRanker() instanceof RankNetMLP);
  }
}
