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

import java.io.IOException;
import java.io.Reader;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

public class LTRResponseHandler {
  protected final LTRResponse response;

  protected LTRResponseHandler(Reader reader) throws IOException{
    ObjectMapper mapper = new ObjectMapper();
    response = mapper.readValue(reader, LTRResponse.class);
  }

  protected Map<String, LTRResponse.Doc[]> getQueryMap(){
    Map<String, LTRResponse.Doc[]> qMap = new HashMap<>();
    LTRResponse.LQuery[] queries = response.results.result.data.queries;
    for(LTRResponse.LQuery lQuery : queries)
      qMap.put(lQuery.query, lQuery.docs);
    return qMap;
  }

  public LTRResponse getResponse() {
    return response;
  }

  public Map<String, LTRResponse.Doc[]> mergeClickRates(Map<String, Map<String, Float>> clickrates){
    Objects.requireNonNull(clickrates);
    Map<String, LTRResponse.Doc[]> qMap = getQueryMap();
    for(Map.Entry<String, LTRResponse.Doc[]> entry : qMap.entrySet()){
      String query = entry.getKey();
      Map<String, Float> queryCR = Objects.requireNonNull(clickrates.get(query), " qid/query mismatch between query and response. \n");
      for(LTRResponse.Doc doc : entry.getValue()){
        Float clickrate = Objects.requireNonNull(queryCR.get(doc.id), "docid mismatch between query and response. \n");
        doc.setClickrate(clickrate);
      }
    }
    return qMap;
  }

}
