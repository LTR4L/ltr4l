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
package org.ltr4l.conversion;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.List;
import java.util.Objects;

public interface LTRModelConverter {
  public SolrLTRModel convert(Reader reader, List<String> featureNames, String modelStore, String featureStore);

  public void write(SolrLTRModel model, Writer writer) throws IOException;

  public static LTRModelConverter get(String algorithm) {
    Objects.requireNonNull(algorithm);
    final String lName = algorithm.toLowerCase();
    switch (lName) {
      case "lambdamart":
        return new LambdaMARTModelConverter();
      case "ranknet":
        return new RankNetModelConverter();
      default:
        throw new IllegalArgumentException("No converter available for " + algorithm);
    }
  }

}
