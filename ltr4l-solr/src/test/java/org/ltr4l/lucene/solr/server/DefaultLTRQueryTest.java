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
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.StringField;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.store.Directory;
import org.apache.lucene.util.IOUtils;
import org.apache.lucene.util.LuceneTestCase;
import org.apache.solr.core.SolrResourceLoader;
import org.junit.Test;
import org.ltr4l.Ranker;

import static org.junit.Assert.*;

import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

public class DefaultLTRQueryTest extends AbstractLTRQueryTestCase {
  SolrResourceLoader loader = new SolrResourceLoader(Paths.get("src/test/resources/collection1/conf"));

  @Test
  public void testDefaultLTRWeightQuery() throws Exception {
    Directory dir = newDirectory();
    RandomIndexWriter w = new RandomIndexWriter(random(), dir, newIndexWriterConfig().setMergePolicy(NoMergePolicy.INSTANCE));

    Document doc = new Document();
    doc.add(new StringField("title", "foo", Field.Store.YES));
    doc.add(new StringField("body", "foo", Field.Store.YES));
    doc.add(new StringField("len", "12", Field.Store.YES));
    w.addDocument(doc);
    w.getReader().close();

    doc = new Document();
    doc.add(new StringField("title", "foo", Field.Store.YES));
    doc.add(new StringField("body", "bar", Field.Store.YES));
    doc.add(new StringField("len", "22", Field.Store.YES));
    w.addDocument(doc);
    w.getReader().close();

    DirectoryReader reader = w.getReader();
    IndexReaderContext context = reader.getContext();

    DefaultLTRModelReader dlmReader = new DefaultLTRModelReader(loader,"ranknet_model.conf");
    Ranker ranker = dlmReader.getRanker();

    DefaultLTRQuery dlQuery = new DefaultLTRQuery(buildFeaturesSpec(
            getTF("TF in title", "title", context, new Term("title", "foo")),
            getTF("TF in body", "body", context, new Term("body", "foo")),
            getTF("TF in name", "name", context, new Term("name", "foo")),
            getIDF("IDF in title", "title", context, new Term("title", "foo")),
            getIDF("IDF in body", "body", context, new Term("body", "foo")),
            getIDF("IDF in name", "name", context, new Term("name", "foo"))), ranker);

    IndexSearcher searcher = new IndexSearcher(reader);
    TopDocs topDocs = searcher.search(dlQuery, 10);
    assertEquals(2, topDocs.totalHits.value);

    // title: foo, body:bar
    assertEquals("bar", searcher.doc(topDocs.scoreDocs[0].doc).get("body"));
    float expectedScore = 0.49607399106025696f;
    assertEquals(expectedScore, topDocs.scoreDocs[0].score, 0.0);

    // title: foo, body:foo
    assertEquals("foo", searcher.doc(topDocs.scoreDocs[1].doc).get("body"));
    expectedScore = 0.4960322976112366f;
    assertEquals(expectedScore, topDocs.scoreDocs[1].score, 0.0);

    IOUtils.close(reader, w, dir);
  }
}
