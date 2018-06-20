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

import java.io.*;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import com.fasterxml.jackson.databind.ObjectMapper;

public class ClickModels {
  private static ClickModels singleton = new ClickModels();

  private ClickModels() { }

  public static ClickModels getInstance() {
    return singleton;
  }

  public static List<ImpressionLog> parseImpressionLog(File file) throws FileNotFoundException {
    if (!file.exists() || !file.canRead() || file.isDirectory()) {
      System.err.println(file.getName() + "is not available or not readable or a directory.");
      return null;
    }

    return parseImpressionLog(new FileInputStream(file));
  }

  public static List<ImpressionLog> parseImpressionLog(InputStream inputStream) {
    List<ImpressionLog> impressionLogList = new ArrayList<ImpressionLog>();
    ObjectMapper objectMapper = new ObjectMapper();

    try {
      Map<String, Object> objectMap = objectMapper.readValue(inputStream, Map.class);

      List<Map<String, Object>> impressionLogs = (List<Map<String, Object>>)objectMap.get("data");
      if (impressionLogs == null) {
        return null;
      }

      for(Map<String, Object> impressionLog : impressionLogs) {
        try {
          impressionLogList.add(new ImpressionLog(
            (String)impressionLog.get("query"),
            (List<String>)impressionLog.get("impressions"),
            (List<String>)impressionLog.get("clicks")));
        } catch (ClassCastException cce) {
          cce.printStackTrace();
        }
      }
    } catch (IOException ioe) {
      ioe.printStackTrace();
      return null;
    }

    return impressionLogList;
  }
}


