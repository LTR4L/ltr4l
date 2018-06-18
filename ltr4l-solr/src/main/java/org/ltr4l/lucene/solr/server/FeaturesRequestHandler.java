package org.ltr4l.lucene.solr.server;

import org.apache.commons.io.IOUtils;
import org.apache.solr.common.SolrException;
import org.apache.solr.common.util.ContentStream;
import org.apache.solr.common.util.SimpleOrderedMap;
import org.apache.solr.handler.RequestHandlerBase;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.response.SolrQueryResponse;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Reader;
import java.lang.invoke.MethodHandles;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class FeaturesRequestHandler extends RequestHandlerBase {

  private static final Logger log = LoggerFactory.getLogger(MethodHandles.lookup().lookupClass());

  Map<Long, FeaturesExtractorManager> managers = new HashMap<Long, FeaturesExtractorManager>();

  /*
  To test the extractor, use curl like this:
    curl -v -H "Accept: application/json" -H "Content-type: application/json" -X POST -d @examples/ltr-queries.json http://localhost:8983/solr/collection1/features?command=extract&conf=ltr_features.conf
   */
  /*
   * available commands:
   *   - /features?command=extract&conf=<json config file name> (async, returns procId)
   *   - /features?command=progress&procId=<procId>
   *   - /features?command=download&procId=<procId>&delete=true
   *   - /features?command=delete&procId=<procId>
   */
  @Override
  public void handleRequestBody(SolrQueryRequest req, SolrQueryResponse rsp) throws Exception {
    SimpleOrderedMap<Object> results = new SimpleOrderedMap<Object>();
    String command = req.getParams().required().get("command");
    results.add("command", command);

    if(command.equals("extract")){
      Iterable<ContentStream> ite = req.getContentStreams();
      if(ite == null){
        throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "no queries found");
      }
      else{
        handleExtract(req, ite, results);
      }
    }
    else if(command.equals("progress")){
      handleProgress(req, results);
    }
    else if(command.equals("download")){
      long procId = req.getParams().required().getLong("procId");
      final boolean delete = req.getParams().getBool("delete", false);
      SimpleOrderedMap<Object> data = download(procId, delete);
      results.add("procId", procId);
      if(data == null){
        FeaturesExtractorManager manager = getManager(procId);
        results.add("done?", manager.isDone());
        results.add("progress", manager.getProgress());
        results.add("result", "the process still runs...");
      }
      else{
        if(delete){
          results.add("deleted", "the process has been removed and the procId is no longer valid");
        }
        results.add("result", data);
      }
    }
    else if(command.equals("delete")){
      handleDelete(req, results);
    }
    else{
      throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, "unknown command " + command);
    }

    rsp.add("results", results);
  }

  public void handleExtract(SolrQueryRequest req, Iterable<ContentStream> ite, SimpleOrderedMap<Object> results)
          throws Exception {
    FeaturesConfigReader fcReader = new FeaturesConfigReader(req.getCore().getResourceLoader(),
            req.getParams().required().get("conf"));
    FeaturesConfigReader.FeatureDesc[] featureDescs = fcReader.getFeatureDescs();
    List<FieldFeatureExtractorFactory> featuresSpec = new ArrayList<FieldFeatureExtractorFactory>();
    for(FeaturesConfigReader.FeatureDesc featureDesc: featureDescs){
      FieldFeatureExtractorFactory dfeFactory = FeaturesConfigReader.loadFactory(featureDesc);
      featuresSpec.add(dfeFactory);
    }
    StringBuilder queries = new StringBuilder();
    for(ContentStream cs: ite){
      Reader reader = cs.getReader();
      try{
        queries.append(IOUtils.toString(reader));
      }
      finally{
        IOUtils.closeQuietly(reader);
      }
    }
    long procId = startExtractor(req, featuresSpec, queries.toString());
    FeaturesExtractorManager manager = getManager(procId);
    results.add("procId", procId);
    results.add("progress", manager.getProgress());
  }

  public void handleProgress(SolrQueryRequest req, SimpleOrderedMap<Object> results) throws Exception {
    long procId = req.getParams().required().getLong("procId");
    FeaturesExtractorManager manager = getManager(procId);
    results.add("procId", procId);
    results.add("done?", manager.isDone());
    results.add("progress", manager.getProgress());
  }

  public void handleDelete(SolrQueryRequest req, SimpleOrderedMap<Object> results) throws Exception {
    long procId = req.getParams().required().getLong("procId");
    delete(procId);
    results.add("procId", procId);
    results.add("result", "the process has been removed and the procId is no longer valid");
  }

  @Override
  public String getDescription() {
    return "Features extraction for NLP4L-LTR";
  }

  public long startExtractor(SolrQueryRequest req, List<FieldFeatureExtractorFactory> featuresSpec, String json) throws Exception {
    // use current server time as the procId
    long procId = System.currentTimeMillis();

    FeaturesExtractorManager manager = new FeaturesExtractorManager(req, featuresSpec, json);
    synchronized(manager) {
      managers.put(procId, manager);
    }

    return procId;
  }

  public FeaturesExtractorManager getManager(long procId){
    FeaturesExtractorManager manager = null;
    synchronized (managers){
      manager = managers.get(procId);
    }
    if(manager == null){
      throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, String.format("no such process (procId=%d)", procId));
    }
    else{
      return manager;
    }
  }

  public void delete(long procId){
    synchronized (managers){
      FeaturesExtractorManager manager = managers.get(procId);
      if(manager == null){
        throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, String.format("no such process (procId=%d)", procId));
      }
      else{
        manager.delete();
        managers.remove(procId);
      }
    }
  }

  public SimpleOrderedMap<Object> download(long procId, boolean delete){
    SimpleOrderedMap<Object> data = null;

    synchronized (managers){
      FeaturesExtractorManager manager = managers.get(procId);
      if(manager == null){
        throw new SolrException(SolrException.ErrorCode.BAD_REQUEST, String.format("no such process (procId=%d)", procId));
      }
      else{
        data = manager.getResult();
        if(delete){
          manager.delete();
          managers.remove(procId);
        }
      }
    }

    return data;
  }
}
