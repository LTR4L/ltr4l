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

import com.fasterxml.jackson.annotation.JsonIgnore;

public class LTRResponse {
  public ResponseHeader responseHeader;
  public Results results;

  public static class ResponseHeader {
    public int QTime;
    public int status;
  }

  public static class Results {
    public String command;
    public long procId;
    public Result result;
  }

  public static class Result {
    public Data data;
  }

  public static class Data {
    public String[] lucene;
    public LQuery[] queries;
  }

  public static class LQuery {
    public Doc[] docs;
    public int qid;
    public String query;
  }

  public static class Doc {
    public double[] lucene;
    public String id;
    @JsonIgnore
    private float clickrate;

    public void setClickrate(float clickrate) {
      this.clickrate = clickrate;
    }

    public double getClickrate() {
      return clickrate;
    }
  }
}
