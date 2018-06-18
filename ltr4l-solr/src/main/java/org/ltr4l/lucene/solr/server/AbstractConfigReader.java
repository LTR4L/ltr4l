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
