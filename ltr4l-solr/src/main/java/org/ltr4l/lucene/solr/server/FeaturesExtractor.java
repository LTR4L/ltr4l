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

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReaderContext;
import org.apache.lucene.index.LeafReaderContext;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.DocIdSetIterator;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.TermQuery;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.util.IOUtils;
import org.apache.solr.request.SolrQueryRequest;
import org.apache.solr.schema.FieldType;

import java.io.File;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.Callable;

public class FeaturesExtractor implements Callable<Integer> {

  private final File featuresFile;
  private final SolrQueryRequest req;
  private final List<FieldFeatureExtractorFactory> featuresSpec;
  private final String idField;
  private final TrainingDataReader.QueryDataDesc[] queryDataDescs;
  private final float totalDocs;
  private float progress = 0;

  public FeaturesExtractor(SolrQueryRequest req, List<FieldFeatureExtractorFactory> featuresSpec, String json, File featuresFile) throws IOException {
    this.featuresFile = featuresFile;
    this.req = req;
    this.featuresSpec = featuresSpec;
    TrainingDataReader tdReader = new TrainingDataReader(json);
    idField = tdReader.getIdField();
    queryDataDescs = tdReader.getQueryDataDescs();
    totalDocs = (float)tdReader.getTotalDocs();
  }

  @Override
  public Integer call() {
    PrintWriter pw = null;
    final boolean _debug = false;
    List<Explanation> _debugExpls = null;
    try {
      pw = new PrintWriter(featuresFile);
      pw.println("{");
      pw.print("  lucene: [");
      int cntFE = 0;
      for(FieldFeatureExtractorFactory factory: featuresSpec){
        if(cntFE > 0){
          pw.printf(" ,\"%s\"", factory.getFeatureName());
        }
        else{
          pw.printf(" \"%s\"", factory.getFeatureName());
        }
        cntFE++;
      }
      pw.println("],");
      pw.println("  queries: [");

      IndexReaderContext context = req.getSearcher().getTopReaderContext();
      int cntQ = 0;
      for (TrainingDataReader.QueryDataDesc queryDataDesc : queryDataDescs) {
        if(cntQ > 0){
          pw.println(",\n    {");
        }
        else{
          pw.println("    {");
        }
        final int qid = queryDataDesc.qid;
        final String qstr = queryDataDesc.queryStr;
        pw.printf("      qid: %d,\n", qid);
        pw.printf("      query: \"%s\",\n", qstr);
        pw.println("      docs: [");
        List<Integer> docIds = new ArrayList<Integer>();
        int cntD = 0;
        for (String key : queryDataDesc.docs) {
          TermQuery idQuery = new TermQuery(new Term(idField, key));
          TopDocs topDocs = req.getSearcher().search(idQuery, 1);
          if(topDocs.scoreDocs.length > 0){
            docIds.add(topDocs.scoreDocs[0].doc);
          }
        }
        Collections.sort(docIds);

        List<LeafReaderContext> leaves = req.getSearcher().getIndexReader().leaves();

        int readerUpto = -1;
        int endDoc = 0;
        int docBase = 0;
        List<FieldFeatureExtractor[]> spec = null;
        Set<Integer> allDocs = null;

        for(int docId: docIds){
          LeafReaderContext readerContext = null;
          while (docId >= endDoc) {
            readerUpto++;
            readerContext = leaves.get(readerUpto);
            endDoc = readerContext.docBase + readerContext.reader().maxDoc();
          }

          if (readerContext != null) {
            // We advanced to another segment:
            docBase = readerContext.docBase;
            spec = new ArrayList<FieldFeatureExtractor[]>();
            allDocs = new HashSet<Integer>();
            for(FieldFeatureExtractorFactory factory: featuresSpec){
              String fieldName = factory.getFieldName();
              FieldType fieldType = req.getSchema().getFieldType(fieldName);
              Analyzer analyzer = fieldType.getQueryAnalyzer();
              factory.init(context, FieldFeatureExtractorFactory.terms(fieldName, qstr, analyzer));
              FieldFeatureExtractor[] extractors = factory.create(readerContext, allDocs);
              spec.add(extractors);
            }
          }
          if(allDocs.size() > 0){
            final List<Integer> aldocs = new ArrayList<Integer>(allDocs);
            Collections.sort(aldocs);
            DocIdSetIterator disi = new DocIdSetIterator() {
              int pos = -1;
              int docId = -1;

              @Override
              public int docID() {
                return docId;
              }

              @Override
              public int nextDoc() throws IOException {
                pos++;
                docId = pos >= aldocs.size() ? NO_MORE_DOCS : aldocs.get(pos);
                return docId;
              }

              @Override
              public int advance(int target) throws IOException {
                while(docId < target){
                  nextDoc();
                }
                return docId;
              }

              @Override
              public long cost() {
                return 0;
              }
            };

            int targetDoc = docId - docBase;
            int actualDoc = disi.docID();
            if (actualDoc < targetDoc) {
              actualDoc = disi.advance(targetDoc);
            }

            if (actualDoc == targetDoc) {
              if(cntD > 0){
                pw.println(",\n        {");
              }
              else{
                pw.println("        {");
              }
              Document luceneDoc = req.getSearcher().doc(docId);
              String idValue = luceneDoc.get(idField);
              pw.printf("          id: \"%s\",\n", idValue);
              // output each lucene
              pw.print("          lucene: [");
              int cntF = 0;
              for(FieldFeatureExtractor[] extractors: spec){
                float feature = 0;
                if(_debug){
                  _debugExpls = new ArrayList<Explanation>();
                }
                for(FieldFeatureExtractor extractor: extractors){
                  feature += extractor.feature(targetDoc);
                  if(_debug){
                    _debugExpls.add(extractor.explain(targetDoc));
                  }
                }
                if(cntF > 0){
                  pw.printf(", %f", feature);
                }
                else{
                  pw.printf(" %f", feature);
                }
                if(_debug){
                  pw.printf(": %s", Explanation.match(feature, "sum of ", _debugExpls));
                }
                cntF++;
              }
              pw.println("]");
              pw.print("        }");   // end of a doc
              cntD++;
            } else {
              // Query did not match this doc, no output
              assert actualDoc > targetDoc;
            }
          }

          incProgress();
        }
        pw.println("\n      ]");  // end of docs

        pw.print("    }");  // end of a query
        cntQ++;
      }

      pw.println("\n  ]");
      pw.println("}");
    }
    catch (IOException e){
      throw new RuntimeException(e);
    }
    finally {
      IOUtils.closeWhileHandlingException(pw);
    }

    req.close();
    return 100;
  }

  public int reportProgress(){
    return getProgress();
  }

  private synchronized void incProgress(){
    progress += 1;
  }

  private synchronized int getProgress(){
    return (int)(progress / totalDocs) * 100;
  }
}
