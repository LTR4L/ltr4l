/**
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

package org.ltr4l;

import java.io.IOException;

import org.ltr4l.query.QuerySet;
import org.ltr4l.trainers.Trainer;

/**
 * LTR Project
 * <p>
 * Please download test datasets from:
 * https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&amp;id=8FEADC23D838BDA8%21107&amp;cid=8FEADC23D838BDA8
 */
public class Main {
  public static void main(String[] args) throws IOException {
    String algorithm = args[0];
    String trainingPath = args[1];
    String validationPath = args[2];
    String configPath = args[3];

    QuerySet trainingSet = QuerySet.create(trainingPath);
    QuerySet validationSet = QuerySet.create(validationPath);

    Trainer trainer = Trainer.TrainerFactory.getTrainer(algorithm, trainingSet, validationSet, configPath);
    long startTime = System.currentTimeMillis();
    trainer.trainAndValidate();
    long endTime = System.currentTimeMillis();
    System.out.println("Took " + (endTime - startTime) + " ms to complete epochs.");
  }
}
