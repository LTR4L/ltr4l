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
package org.ltr4l.query;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RankedDocs {
  private final List<Document> rankedDocs;

  public RankedDocs(List<Document> unrankedDocs){
    rankedDocs = new ArrayList<>(unrankedDocs);
    rankedDocs.sort((doca, docb) -> Integer.compare(docb.getLabel(), doca.getLabel()));
  }

  public int getLabel(int i){
    return rankedDocs.get(i).getLabel();
  }

  public List<Document> getRankedDocs(){
    return Collections.unmodifiableList(rankedDocs);
  }

  public int size(){
    return rankedDocs.size();
  }

  public Document get(int i){
    return rankedDocs.get(i);
  }

}
