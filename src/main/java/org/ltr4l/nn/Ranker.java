package org.ltr4l.nn;

import org.ltr4l.query.Document;

import java.util.Properties;

/**
 * Copyright [yyyy] [name of copyright owner]
 * <p>
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * <p>
 * http://www.apache.org/licenses/LICENSE-2.0
 * <p>
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
public abstract class Ranker {
  protected static final String DEFAULT_MODEL_FILE = "model.txt";

  abstract void writeModel(Properties prop, String file);
  protected void writeModel(Properties prop){
    writeModel(prop, DEFAULT_MODEL_FILE);
  }

  abstract double predict(Document document);

}
