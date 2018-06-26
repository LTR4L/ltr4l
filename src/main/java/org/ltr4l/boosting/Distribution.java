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
package org.ltr4l.boosting;

import org.ltr4l.query.Document;
import org.ltr4l.query.RankedDocs;

import java.util.List;

public abstract class Distribution {
  protected double normFactor;

  Distribution(){
    normFactor = 1d;
  }

  protected abstract void initialize(List<RankedDocs> queries);
  protected abstract double updateQuery(WeakLearner wl, int qid, List<Document> queryDocs);
  protected abstract void normalize(double normFactor);

  public void update(WeakLearner wl, List<RankedDocs> queries ){
    double newNormFactor = 0d;
    for(int qid = 0; qid < queries.size(); qid++){
      newNormFactor += updateQuery(wl, qid, queries.get(qid));
    }
    normalize(newNormFactor);
  }


  public static abstract class D2 extends Distribution{
    protected final double[][] dist;

    public D2(List<RankedDocs> queries){
      dist = new double[queries.size()][];
      initialize(queries);
    }

    @Override
    protected void normalize(double normFactor){
      for(int qid = 0; qid < dist.length; qid++){
        for(int i = 0; i < dist[qid].length; i++)
          dist[qid][i] /= normFactor;
      }
      this.normFactor = normFactor;
    }

    public double[][] getFullDist(){
      return dist;
    }
    public double[] getQueryDist(int i){
      return dist[i];
    }
    public void setQueryDist(int i, double[] newDist){
      dist[i] = newDist;
    }
  }

  /**
   * 3-Dimensional distribution used for boosting. R
   * means that RankedDocs is required (Documents must be
   * sorted in order of descending label)
   */
  public static abstract class D3R extends Distribution{
    protected final double[][][] dist;

    public D3R(List<RankedDocs> rQueries){
      dist = new double[rQueries.size()][][];
      initialize(rQueries);
    }

    protected void normalize(double normFactor){
      for(int qid = 0; qid < dist.length; qid++){
        for(int i = 0; i < dist[qid].length; i++)
          for(int j = 0; j < dist[qid][i].length; j++) //Note: could be sped up
            dist[qid][i][j] /= normFactor;
      }
      this.normFactor = normFactor;
    }
  }
}
