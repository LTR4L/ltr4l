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

import java.io.*;
import java.util.Map;

public class DefaultLTRModelReader extends AbstractConfigReader {
  protected String solrHome;
  protected String algorithm;
  protected Reader reader;

  public DefaultLTRModelReader(String fileName) throws IOException {
    this(null, fileName);
  }

  public DefaultLTRModelReader(SolrResourceLoader loader, String fileName) throws IOException {
    super(loader, fileName);
    String algorithm = (String)((Map)configMap.get("config")).get("algorithm");

    if (loader == null) {
      loader = new SolrResourceLoader();
    }
    solrHome = loader.locateSolrHome().toString();

    // To avoid opening model files every time when getting ranker.
    StringBuilder sb = new StringBuilder();
    try (InputStream is = new FileInputStream(solrHome + "/" + fileName);
         InputStreamReader iReader = new InputStreamReader(is, "UTF-8")){
      char buf[] = new char[8192];
      int numRead;
      while(0 <= (numRead = reader.read(buf))) {
        sb.append(buf, 0, numRead);
      }
    } catch (IOException ioe) {
      throw ioe;
    }

    reader = new CharArrayReader(sb.toString().toCharArray());
  }

  public Ranker getRanker() throws IOException{
    return Ranker.RankerFactory.getFromModel(algorithm, reader);
  }
}
