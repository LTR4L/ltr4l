@echo off
title Convert-model batch script
dir /b LTR4L-*.jar > script_temp_file.txt
set /p JAR_FILE= < script_temp_file.txt
java -cp "%JAR_FILE%" org.ltr4l.conversion.Convert %*
del script_temp_file.txt