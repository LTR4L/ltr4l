package org.ltr4l.tools;

import java.io.BufferedWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.file.StandardOpenOption;

public class Report {
  public static void report(int iter, double ndcg, double tloss, double vloss) {
    report("data/data.csv", iter, ndcg, tloss, vloss);
  }

  //Change design so that bufferedWriter is only opened once?
  public static void report(String filepath, int iter, double ndcg, double tloss, double vloss) {
    try {
      if (!Files.exists(Paths.get(filepath)))
        Files.createFile(Paths.get(filepath));
      BufferedWriter bw = Files.newBufferedWriter(Paths.get(filepath), StandardOpenOption.APPEND);
      bw.append(iter + "," + ndcg + "," + tloss + "," + vloss);
      bw.newLine();
      bw.close();
    } catch (IOException e) {
      e.printStackTrace();
    }

  }
}
