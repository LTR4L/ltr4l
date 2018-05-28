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
package org.ltr4l.click;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;

public class ClickModelConverter {

  public static void getCMQuery(InputStream inputStream, OutputStream outputStream) throws IOException{
    List<ImpressionLog> impressionLogList = ClickModels.getInstance().parseImpressionLog(inputStream);
    ClickModelAnalyzer clickModelAnalyzer = new ClickModelAnalyzer();
    Map<String, Map<String, Float>> clickRates = clickModelAnalyzer.calcClickRate(impressionLogList);
    CMQueries cmq = new CMQueries(clickRates);
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(outputStream, cmq);
  }

  public static void getCMQuery(InputStream inputStream) throws IOException{
    getCMQuery(inputStream, new ByteArrayOutputStream());
  }

  public static class CMQueries {
    public String idField;
    public List<CMQuery> queries;

    public CMQueries(Map<String, Map<String, Float>> clickRates){
      idField = "url";
      queries = new ArrayList<>();
      int qid = 0; //TODO: ok to start qid from 0?
      for(String query : clickRates.keySet()){
        Map<String, Float> values = clickRates.get(query);
        Set<String> docSet = values.keySet();
        String[] docs = docSet.toArray(new String[docSet.size()]);
        queries.add(new CMQuery(qid, query, docs));
        qid++;
      }
    }

    public static class CMQuery{
      public int qid;
      public String query;
      public String[] docs;

      public CMQuery(int qid, String query, String[] docs){
        this.qid = qid;
        this.query = query;
        this.docs = docs;
      }
    }
  }

}
