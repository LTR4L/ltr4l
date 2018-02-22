package org.ltr4l;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

import org.ltr4l.query.QuerySet;
import org.ltr4l.tools.Config;
import org.ltr4l.trainers.Trainer;

/**
 * LTR Project
 * <p>
 * Please download test datasets from:
 * https://onedrive.live.com/?authkey=%21ACnoZZSZVfHPJd0&id=8FEADC23D838BDA8%21107&cid=8FEADC23D838BDA8
 */
public class Main {
  public static void main(String[] args) throws IOException {
    String trainingPath = args[0];
    String validationPath = args[1];
    String configPath = args[2];

    prepareDataFile(null);
    QuerySet trainingSet = QuerySet.create(trainingPath);
    QuerySet validationSet = QuerySet.create(validationPath);
    Config configs = Config.get(configPath);

    String algorithm = configs.getName();
    Trainer trainer = Trainer.TrainerFactory.getTrainer(algorithm, trainingSet, validationSet, configs);
    long startTime = System.currentTimeMillis();
    trainer.trainAndValidate();
    long endTime = System.currentTimeMillis();
    System.out.println("Took " + (endTime - startTime) + " ms to complete epochs.");
  }

  private static void prepareDataFile(String dataSavePath) throws IOException {
    BufferedWriter bw;
    if (dataSavePath != null)
      bw = Files.newBufferedWriter(Paths.get(dataSavePath), StandardOpenOption.APPEND, StandardOpenOption.CREATE_NEW);
    else {
      bw = Files.newBufferedWriter(Paths.get("data/data.csv")/*, StandardOpenOption.APPEND, StandardOpenOption.CREATE_NEW*/);
    }
    bw.append(",NDCG@10,tr_loss,va_loss");
    bw.newLine();
    bw.close();
  }


}
