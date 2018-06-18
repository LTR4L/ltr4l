package org.ltr4l.lucene.solr.server;

import java.io.IOException;
import java.util.List;
import java.util.Map;

public class TrainingDataReader extends AbstractConfigReader {

  private final QueryDataDesc[] queryDataDescs;
  private final String idField;
  private final int totalDocs;

  public TrainingDataReader(String content) throws IOException {
    super(content);
    idField = (String)configMap.get("idField");
    List<? extends Map> queriesData = (List)configMap.get("queries");
    queryDataDescs = new QueryDataDesc[queriesData.size()];
    int i = 0;
    int totalDocs = 0;
    for(Map queryData: queriesData){
      int qid = (Integer)queryData.get("qid");
      String queryStr = (String)queryData.get("query");
      List<String> docs = (List)queryData.get("docs");
      totalDocs += docs.size();
      queryDataDescs[i++] = new QueryDataDesc(qid, queryStr, docs.toArray(new String[docs.size()]));
    }
    this.totalDocs = totalDocs;
  }

  public String getIdField(){
    return idField;
  }

  public QueryDataDesc[] getQueryDataDescs(){
    return queryDataDescs;
  }

  public int getTotalDocs(){
    return totalDocs;
  }

  public static class QueryDataDesc {
    public final int qid;
    public final String queryStr;
    public final String[] docs;
    public QueryDataDesc(int qid, String queryStr, String[] docs){
      this.qid = qid;
      this.queryStr = queryStr;
      this.docs = docs;
    }

    @Override
    public String toString(){
      StringBuilder sb = new StringBuilder();
      sb.append("qid=").append(qid).append(",query=").append(queryStr).append(",docs=[");
      for(int i = 0; i < docs.length; i++){
        if(i > 0) sb.append(',');
        sb.append(docs[i]);
      }
      sb.append("]");
      return sb.toString();
    }
  }
}
