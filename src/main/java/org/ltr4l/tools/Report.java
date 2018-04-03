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

package org.ltr4l.tools;

import java.io.FileOutputStream;
import java.io.IOException;
import java.io.PrintWriter;

public class Report {

  private static final String DEFAULT_REPORT_FILE = "report/report.csv";
  private final PrintWriter pw;
  private final String file;

  public static Report getReport(){
    return getReport(DEFAULT_REPORT_FILE);
  }

  public static Report getReport(String file){
    file = (file == null || file.isEmpty()) ? DEFAULT_REPORT_FILE : file;
    try {
      return new Report(file);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  public static Report getReport(String file, String header){
    file = (file == null || file.isEmpty()) ? DEFAULT_REPORT_FILE : file;
    try {
      return new Report(file, header);
    } catch (IOException e) {
      throw new RuntimeException(e);
    }
  }

  private Report(String file, String header) throws IOException{ //TODO: eval method should be mutable
    this.file = file;
    pw = new PrintWriter(new FileOutputStream(file));
    // TODO: eval method should be mutable
    pw.println(header);  // header for CSV file
  }

  private Report(String file) throws IOException {
    this.file = file;
    pw = new PrintWriter(new FileOutputStream(file));
    // TODO: eval method should be mutable
    pw.println(",evaluation,tr_loss,va_loss");  // header for CSV file
  }

  public void log(int iter, double eval, double tloss, double vloss){
    // TODO: eval method should be mutable
    System.out.printf("%d tr_loss: %f va_loss: %f evaluation: %f\n", iter, tloss, vloss, eval);
    pw.printf("%d,%f,%f,%f\n", iter, eval, tloss, vloss);
  }

  public void log(double eval){
    System.out.printf("Evaluation score: %f\n", eval);
    pw.printf("%f", eval);
  }

  public void close(){
    if(pw != null) pw.close();
  }

  public String getReportFile(){
    return file;
  }
}
