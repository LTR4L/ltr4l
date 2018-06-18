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

import java.io.IOException;
import java.lang.reflect.Constructor;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FeaturesConfigReader extends AbstractConfigReader {

  private final FeatureDesc[] featureDescs;
  private final Map<String, FeatureDesc> fdMap;

  public FeaturesConfigReader(String fileName) throws IOException {
    this(null, fileName);
  }

  public FeaturesConfigReader(SolrResourceLoader loader, String fileName) throws IOException {
    super(loader, fileName);
    List<? extends Map> featuresConfig = (List)configMap.get("features");
    featureDescs = new FeatureDesc[featuresConfig.size()];
    fdMap = new HashMap<String, FeatureDesc>();
    int i = 0;
    for(Map featureConfig: featuresConfig){
      String name = (String)featureConfig.get("name");
      String klass = (String)featureConfig.get("class");
      String param = ((Map<String, String>)featureConfig.get("params")).get("field");  // TODO: consider multiple parameters
      featureDescs[i] = new FeatureDesc(name, klass, param);
      fdMap.put(name, featureDescs[i]);
      i++;
    }
  }

  public FeatureDesc[] getFeatureDescs(){
    return featureDescs;
  }

  public FeatureDesc getFeatureDesc(String name){
    return fdMap.get(name);
  }

  public static FieldFeatureExtractorFactory loadFactory(FeaturesConfigReader.FeatureDesc featureDesc){
    ClassLoader loader = Thread.currentThread().getContextClassLoader();
    try {
      Class<? extends FieldFeatureExtractorFactory> cls = (Class<? extends FieldFeatureExtractorFactory>) loader.loadClass(featureDesc.klass);
      Class<?>[] types = {String.class, String.class};
      Constructor<? extends FieldFeatureExtractorFactory> constructor;
      constructor = cls.getConstructor(types);
      return constructor.newInstance(featureDesc.name, featureDesc.param);
    } catch (Exception e) {
      throw new RuntimeException(e);
    }
  }

  public static class FeatureDesc {
    public final String name;
    public final String klass;
    public final String param;   // TODO: change param to params as a type of Map<key, value>
    public FeatureDesc(String name, String klass, String param){
      this.name = name;
      this.klass = klass;
      this.param = param;
    }

    @Override
    public String toString(){
      StringBuilder sb = new StringBuilder();
      sb.append("name=").append(name).append(",class=").append(klass).append(",params=").append(param);
      return sb.toString();
    }
  }
}
