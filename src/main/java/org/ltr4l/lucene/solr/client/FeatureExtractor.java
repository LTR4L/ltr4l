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

package org.ltr4l.lucene.solr.client;

import com.fasterxml.jackson.databind.ObjectMapper;
import org.apache.http.HttpEntity;
import org.apache.http.HttpResponse;
import org.apache.http.client.HttpClient;
import org.apache.http.client.methods.HttpGet;
import org.apache.http.client.methods.HttpPost;
import org.apache.http.entity.StringEntity;
import org.apache.http.impl.client.HttpClients;
import org.apache.http.util.EntityUtils;
import org.ltr4l.click.CMQueryHandler;
import org.ltr4l.click.LTRResponse;
import org.ltr4l.click.LTRResponseHandler;
import org.ltr4l.click.ClickRateClassifier;

import java.io.InputStreamReader;
import java.util.*;

public class FeatureExtractor {

  private final CMQueryHandler cmQueryHandler;
  private LTRResponseHandler ltrResponseHandler;
  private final String url;
  private final String confName;
  private final String idField;
  private final long extractionTimeout;
  private String procId;

  public FeatureExtractor(String url, String confName, CMQueryHandler cmQueryHandler) {
    this(url, confName, cmQueryHandler, "id", 10000L);
  }

  public FeatureExtractor(String url, String confName, CMQueryHandler cmQueryHandler, Long extractionTimeout) {
    this(url, confName, cmQueryHandler, "id", extractionTimeout);
  }

  public FeatureExtractor(String url, String confName, CMQueryHandler cmQueryHandler, String idField) {
    this(url, confName, cmQueryHandler, idField, 10000L);
  }

  public FeatureExtractor(String url, String confName, CMQueryHandler cmQueryHandler, String idField, Long extractionTimeout) {
    this.url = url;
    this.confName = confName;
    this.cmQueryHandler = cmQueryHandler;
    this.idField = idField;
    this.extractionTimeout = extractionTimeout;
  }

  public void execute() throws Exception {
    if( url == null || url.equals(""))
      return;

    postTrainingData();
    if(isFinished()) {
      download();
    } else {
      System.err.println("Could not finish feature extraction.\nPlease set longer extraction timeout period or confirm the url and conf name are valid.");
    }
  }

  private void postTrainingData() throws Exception {
    HttpClient httpClient = HttpClients.createDefault();

    CMQueryHandler.CMQueries cmQueries = new CMQueryHandler.CMQueries(cmQueryHandler.getClickRates(), idField);
    String url = this.url + "?command=extract&conf=" + confName + "&wt=json";
    StringEntity trainingJson = new StringEntity(cmQueries.toString());
    HttpPost httpPost = new HttpPost(url);
    httpPost.setEntity(trainingJson);
    httpPost.addHeader("Content-type", "application/json");
    httpPost.addHeader("Accept", "application/json");

    HttpResponse response = httpClient.execute(httpPost);
    HttpEntity entity = response.getEntity();

    if (entity != null) {
      ObjectMapper mapper = new ObjectMapper();
      Map<String, Object> entityMap = mapper.readValue(EntityUtils.toString(entity), Map.class);
      Map<String, Object> results = (Map<String, Object>) entityMap.get("results");
      procId = results == null ? null : String.valueOf(results.get("procId"));
    }
  }

  private boolean isFinished() throws Exception {
    String url = this.url + "?command=progress&procId=" + procId + "&wt=json";
    HttpGet httpGet = new HttpGet(url);
    HttpClient httpClient = HttpClients.createDefault();
    int progress = 0;
    long startTime = System.nanoTime();
    long duration = 0;

    //TODO: smarter code
    while (progress < 100 && duration < extractionTimeout * 10000) {
      Thread.sleep(1000);
      HttpResponse response = httpClient.execute(httpGet);
      HttpEntity entity = response.getEntity();
      if (entity != null) {
        ObjectMapper mapper = new ObjectMapper();
        Map<String, Object> entityMap = mapper.readValue(EntityUtils.toString(entity), Map.class);
        Map<String, Object> results = (Map<String, Object>) entityMap.get("results");
        progress = results == null ? 0 : (Integer)results.get("progress");
      } else {
        return false;
      }
      duration = System.nanoTime() - startTime;
    }
    return progress == 100;
  }

  private void download() throws Exception {
    String url = this.url + "?command=download&procId=" + procId + "&wt=json";

    HttpClient httpClient = HttpClients.createDefault();

    HttpGet httpGet = new HttpGet(url);
    HttpResponse response = httpClient.execute(httpGet);

    InputStreamReader inputStreamReader = new InputStreamReader(response.getEntity().getContent());
    ltrResponseHandler = new LTRResponseHandler(inputStreamReader);
  }

  public Map<String, LTRResponse.Doc[]> getTrainingData() {
    return ltrResponseHandler == null ? null : ltrResponseHandler.mergeClickRates(cmQueryHandler.getClickRates());
  }

  //TODO: Large training data may cause OOM, we should parse & write Doc into output file one by one.
  public String getMSFormatTrainingData(String borderListStr) {
    if (ltrResponseHandler == null)
      return null;

    ClickRateClassifier crc = new ClickRateClassifier(borderListStr);

    Map<String, LTRResponse.Doc[]> trainingData = ltrResponseHandler.mergeClickRates(cmQueryHandler.getClickRates());
    StringBuilder sb = new StringBuilder();

    long qid = 0;
    for (Map.Entry<String, LTRResponse.Doc[]> entry : trainingData.entrySet()) {
      LTRResponse.Doc[] docs = entry.getValue();
      for (LTRResponse.Doc doc : docs) {
        sb.append(String.valueOf(crc.classify(doc.getClickrate())) + " " + "qid:" + String.valueOf(qid));
        double[] features = doc.features;
        int len = features.length;
        for (int i = 0; i < len; i++) {
          sb.append(" " + String.valueOf(i+1) + ":" + String.valueOf(features[i]));
        }
        sb.append(" #docid = " + doc.id + "\n");
      }
      qid++;
    }
    return sb.toString();
  }

}
