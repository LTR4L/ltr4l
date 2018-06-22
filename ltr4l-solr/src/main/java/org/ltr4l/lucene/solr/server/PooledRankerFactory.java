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

import org.apache.commons.pool2.PooledObject;
import org.apache.commons.pool2.PooledObjectFactory;
import org.apache.commons.pool2.impl.DefaultPooledObject;
import org.ltr4l.Ranker;

import java.io.IOException;

public class PooledRankerFactory implements PooledObjectFactory<Ranker> {
  private NeuralNetworkModelReader nnmReader;

  public PooledRankerFactory(NeuralNetworkModelReader nnmReader) {
    this.nnmReader = nnmReader;
  }

  public PooledObject<Ranker> makeObject() {
    Ranker ranker;
    try {
      ranker = nnmReader.getRanker();
    } catch (IOException ioe) {
      ioe.printStackTrace();
      return null;
    }
    return new DefaultPooledObject<>(ranker);
  }

  public void destroyObject(PooledObject<Ranker> ranker) throws Exception { }

  public boolean validateObject(PooledObject<Ranker> ranker) {
    return ranker != null;
  }

  public void activateObject(PooledObject<Ranker> ranker) throws Exception { }

  public void passivateObject(PooledObject<Ranker> ranker) throws Exception { }

}
