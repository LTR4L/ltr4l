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

import com.fasterxml.jackson.core.JsonParser;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.lucene.util.IOUtils;
import org.apache.solr.common.util.SimpleOrderedMap;
import org.apache.solr.request.SolrQueryRequest;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class FeaturesExtractorManager {

  private final File featuresFile;
  private final FeaturesExtractor extractor;
  private final ExecutorService executor;
  private final Future<Integer> future;

  public FeaturesExtractorManager(SolrQueryRequest req, List<FieldFeatureExtractorFactory> featuresSpec, String json) throws IOException {
    featuresFile = File.createTempFile("features-", ".json");
    extractor = new FeaturesExtractor(req, featuresSpec, json, featuresFile);
    executor = Executors.newSingleThreadExecutor();
    future = executor.submit(extractor);
    executor.shutdown();
  }

  public FeaturesExtractor getExtractor(){
    return extractor;
  }

  public int getProgress(){
    return extractor.reportProgress();
  }

  public boolean isDone(){
    return future.isDone();
  }

  public void delete(){
    if(featuresFile != null){
      featuresFile.delete();
    }
  }

  public SimpleOrderedMap<Object> getResult(){
    if(future.isDone()){
      SimpleOrderedMap<Object> result = new SimpleOrderedMap<Object>();
      Reader r = null;
      try{
        r = new FileReader(featuresFile);
        ObjectMapper mapper = new ObjectMapper();
        mapper.configure(JsonParser.Feature.ALLOW_UNQUOTED_FIELD_NAMES, true);
        Map<String, Object> json = mapper.readValue(featuresFile, Map.class);
        result.add("data", parseData(json));
        return result;
      } catch (IOException e) {
        throw new RuntimeException(e);
      } finally{
        IOUtils.closeWhileHandlingException(r);
      }
    }
    else return null;
  }

  SimpleOrderedMap<Object> parseData(Map data){
    SimpleOrderedMap<Object> result = new SimpleOrderedMap<Object>();
    result.add("featuresSet", (List<String>)data.get("featuresSet"));
    result.add("queries", parseQueries((List<Map>)data.get("queries")));
    return result;
  }

  List<Object> parseQueries(List<? extends Map> queries){
    List<Object> result = new ArrayList<Object>();
    for(Map q: queries){
      result.add(parseQ(q));
    }
    return result;
  }

  SimpleOrderedMap<Object> parseQ(Map q){
    SimpleOrderedMap<Object> result = new SimpleOrderedMap<Object>();
    result.add("qid", (Integer)q.get("qid"));
    result.add("query", (String)q.get("query"));
    result.add("docs", parseDocs((List<Map>)q.get("docs")));
    return result;
  }

  List<Object> parseDocs(List<? extends Map> docs){
    List<Object> result = new ArrayList<Object>();
    for(Map doc: docs){
      result.add(parseDoc(doc));
    }
    return result;
  }

  SimpleOrderedMap<Object> parseDoc(Map doc){
    SimpleOrderedMap<Object> result = new SimpleOrderedMap<Object>();
    result.add("id", (String)doc.get("id"));
    result.add("features", (List<Double>)doc.get("features"));
    return result;
  }
}
