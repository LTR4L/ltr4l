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

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.util.IOUtils;
import org.apache.solr.core.SolrResourceLoader;

import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;

public abstract class AbstractConfigReader {

  protected static Map configMap = null;

  public AbstractConfigReader(SolrResourceLoader loader, String fileName) throws IOException {
    this(loader, fileName, null);
  }

  public AbstractConfigReader(SolrResourceLoader loader, String fileName, String path) throws IOException {
    configMap = load(loader, fileName);
    configMap = path == null ? configMap : (Map)configMap.get(path);
  }

  public AbstractConfigReader(String content) throws IOException {
    if(content != null) {
          System.err.println("content: " + content);

      InputStream is = new ByteArrayInputStream(content.getBytes());
      try {
        ObjectMapper mapper = new ObjectMapper();
        configMap = mapper.readValue(is, Map.class);
      } finally {
        IOUtils.closeWhileHandlingException(is);
      }
    }
  }

  public static Map load(SolrResourceLoader loader, String fileName) throws IOException {
    if(loader == null)
      loader = new SolrResourceLoader();
    InputStream is = null;
    try {
      ObjectMapper mapper = new ObjectMapper();
      is = loader.openResource(fileName);
      return mapper.readValue(is, Map.class);
    } finally {
      IOUtils.closeWhileHandlingException(is);
    }
  }
}
