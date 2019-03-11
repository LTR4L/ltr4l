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

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import org.ltr4l.boosting.Ensemble;
import org.ltr4l.boosting.RegressionTree;

import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

public class LambdaMARTModelConverter implements LTRModelConverter {

  @Override
  public final void write(SolrLTRModel model, Writer writer) throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    mapper.enable(SerializationFeature.INDENT_OUTPUT);
    mapper.writeValue(writer, model);
  }

  @Override
  public final SolrLTRModel convert(Reader reader, List<String> features) {
    SolrLTRModel solrModel = new SolrLTRModel();
    solrModel.clazz = "org.apache.solr.ltr.model.MultipleAdditiveTreesModel";
    solrModel.name = "lambdamartmodel";
    List<SolrLTRModel.Feature> features1 = features.stream()
        .map(SolrLTRModel.Feature::new)
        .collect(Collectors.toCollection(ArrayList::new));
    solrModel.features = features1;
    solrModel.params = new HashMap<>();
    List<MARTree> trees = new ArrayList<>();
    solrModel.params.put("trees", trees);
    try {
      ObjectMapper mapper = new ObjectMapper();
      Ensemble.SavedModel ltr4lmodel = mapper.readValue(reader, Ensemble.SavedModel.class);
      assert(ltr4lmodel.config.algorithm.equals("LambdaMart"));
      for (RegressionTree.SavedModel ltr4lTree : ltr4lmodel.treeModels) {
        final Map<Integer, LTR4Lnode> idxNodeMap = new HashMap<>();
        int max = 0;
        for (int i = 0; i < ltr4lTree.leafIds.size(); i++) {
          int idx = ltr4lTree.leafIds.get(i);
          if (idx > max)
            max = idx;
          int featId = ltr4lTree.featureIds.get(i);
          double thresh = ltr4lTree.thresh.get(i);
          double score = ltr4lTree.scores.get(i);
          idxNodeMap.put(idx, new LTR4Lnode(idx, featId, thresh, score));
        }
        //Build tree
        MARTree tree = new MARTree();
        trees.add(tree);
        LTR4Lnode lroot = idxNodeMap.get(0);
        MARTnode root = new MARTnode(lroot.leafId, lroot.threshold, features.get(lroot.featureId));
        tree.root = root;
        root.fill(idxNodeMap, features);
      }
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
    return solrModel;
  }



  public static class MARTree {
    public final double weight = 1.0d;
    public MARTnode root;

  }

  public static class MARTnode {
    public String feature;
    public double threshold;
    public Object left;
    public Object right;
    @JsonIgnore
    public final int idx;

    MARTnode(int idx, double threshold, String feature) {
      this.feature = feature;
      this.threshold = threshold;
      this.idx = idx;
    }

    @JsonIgnore
    Object[] getLeaves() {
      return new Object[] {left, right};
    }

    @JsonIgnore
    void fill(Map<Integer, LTR4Lnode> ltr4lNodes, List<String> features) {
      addLeft(ltr4lNodes.get(2 * idx + 1), features);
      addRight(ltr4lNodes.get(2 * idx + 2), features);
      for (Object leaf : getLeaves()) {
        if (leaf instanceof MARTnode)
          ((MARTnode) leaf).fill(ltr4lNodes, features);
      }
    }

    @JsonIgnore
    void addLeft(LTR4Lnode lnode, List<String> features) {
      if (lnode.featureId == -1)
        left = new Terminal(lnode.score);
      else
        left = new MARTnode(lnode.leafId, lnode.threshold, features.get(lnode.featureId));
    }

    @JsonIgnore
    void addRight(LTR4Lnode rnode, List<String> features) {
      if (rnode.featureId == -1)
        right = new Terminal(rnode.score);
      else
        right = new MARTnode(rnode.leafId, rnode.threshold, features.get(rnode.featureId));
    }

  }

  public static class Terminal {
    public double value;

    Terminal(double value) {
      this.value = value;
    }
  }

  private static class LTR4Lnode {
    private int leafId;
    private int featureId;
    private double threshold;
    private double score;

    LTR4Lnode(int leafId, int featureId, double threshold, double score) {
      this.leafId = leafId;
      this.featureId = featureId;
      this.threshold = threshold;
      this.score = score;
    }
  }

}
