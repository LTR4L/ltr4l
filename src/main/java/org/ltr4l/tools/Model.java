package org.ltr4l.tools;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Arrays;
import java.util.List;

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
public class Model {
  private static final String DEFAULT_MODEL_FILE = "model.txt";
  private PrintWriter pw;

  private Model(String file) throws IOException {
    pw = new PrintWriter(new FileOutputStream(file));
  }

  public static Model getModel(String file) {
    try {
      return new Model(file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static Model getModel(){
    return getModel(DEFAULT_MODEL_FILE);
  }

  public void log(double[] weights){
    pw.println(Arrays.toString(weights));
  }

  public void log(List<List<List<Double>>> weights) {
    pw.println(weights);
  }

  public void close(){
    if(pw != null) pw.close();
  }

}
