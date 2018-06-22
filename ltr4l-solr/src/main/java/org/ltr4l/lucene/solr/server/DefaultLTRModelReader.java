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
import org.ltr4l.Ranker;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.Map;

public class DefaultLTRModelReader extends AbstractConfigReader {
  protected static String solrHome;
  protected static String fileName;

  public DefaultLTRModelReader(String fileName) throws IOException {
    this(null, fileName);
  }

  public DefaultLTRModelReader(SolrResourceLoader loader, String fileName) throws IOException {
    super(loader, fileName);
    this.fileName = fileName;
  }

  public Ranker getRanker() throws IOException{
    String algorithm = (String)((Map)configMap.get("config")).get("algorithm");
    SolrResourceLoader loader = new SolrResourceLoader();
    solrHome = loader.locateSolrHome().toString();
    Reader reader = new FileReader(solrHome + "/" + fileName);
    return Ranker.RankerFactory.getFromModel(algorithm, reader);
  }
}
