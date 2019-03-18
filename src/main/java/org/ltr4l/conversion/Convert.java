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

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

public class Convert {
  public static final String DEFAULT_OUTPUT_DIR = "solr-models/";
  public static void main(String[] args) throws IOException {
    if (args.length < 3) {
      throw new IllegalArgumentException("Please provide arguments as follows: <algorithmName> <LTR4LmodelPath> <featuresKeyPath> <solr-model-output>. \n" +
          "Model-output is an optional parameter, and by default will be saved in solr-models/*-smodel.json");
    }
    String algorithm = args[0];
    String path = args[1];
    String featuresPath = args[2];
    String output = args.length >= 4 ? args[3] : DEFAULT_OUTPUT_DIR + algorithm + "-smodel.json";

    LTRModelConverter converter = LTRModelConverter.get(algorithm);
    BufferedReader featuresReader = Files.newBufferedReader(Paths.get(featuresPath));
    String line = featuresReader.readLine();
    while (line != null && !line.startsWith("1"))
      line = featuresReader.readLine();
    if (line == null)
      throw new IllegalArgumentException("File did not contain a line containing features in LETOR format (1:feat1 2:feat2 ...)!");
    featuresReader.close();
    String[] feats = line.split(" ");
    List<String> features = Arrays.stream(feats).map(f -> f.split(":")[1])
        .collect(Collectors.toCollection(ArrayList::new));

    Reader reader = Files.newBufferedReader(Paths.get(path));
    SolrLTRModel model = converter.convert(reader, features);
    reader.close();

    File file = new File(output);
    if (file.getParentFile() != null)
      file.getParentFile().mkdirs();

    Writer writer = Files.newBufferedWriter(Paths.get(output));
    converter.write(model, writer);
  }
}
