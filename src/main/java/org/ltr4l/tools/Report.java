package org.ltr4l.tools;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

public class Report {

  private static final String DEFAULT_REPORT_FILE = "report.csv";
  private final PrintWriter pw;

  public static Report getReport(){
    return getReport(DEFAULT_REPORT_FILE);
  }

  public static Report getReport(String file){
    try {
      return new Report(file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private Report(String file) throws IOException {
    pw = new PrintWriter(new FileOutputStream(file));
    pw.println(",NDCG@10,tr_loss,va_loss");  // header for CSV file
  }

  public void log(int iter, double ndcg, double tloss, double vloss){
    System.out.printf("%d tr_loss: %f va_loss: %f ndcg@10: %f\n", iter, tloss, vloss, ndcg);
    pw.printf("%d,%f,%f,%f\n", iter, ndcg, tloss, vloss);
  }

  public void close(){
    if(pw != null) pw.close();
  }
}
