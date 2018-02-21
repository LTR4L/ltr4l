## Project

Implementation of some LTR Algorithms.

## Requirements

You will need to install the following:

1. Java 8+

2. Apache Ant 1.8+ (Not required to run, however the following instructions will assume Apache Ant has been installed.)

## Data

You can download OHSUMED, LETOR MQ2007, LETOR MQ2008 from [this web page](https://www.microsoft.com/en-us/research/project/letor-learning-rank-information-retrieval/).

**Dataset Descriptions**

Each row is a query-document pair. The first column is relevance label of this pair, the second column is query id, the following columns are features, and the end of the row is comment about the pair, including id of the document. The larger the relevance label, the more relevant the query-document pair. A query-document pair is represented by a 46-dimensional feature vector. Here are several example rows from MQ2007 dataset:
```
====================================================

2 qid:10032 1:0.056537 2:0.000000 3:0.666667 ... 46:0.076923 #docid=GX029-35-5894638 inc=0.02 prob=0.13984

0 qid:10032 1:0.279152 2:0.000000 3:0.000000 ... 46:1.000000 #docid=GX030-77-6315042 inc=0.71 prob=0.34136

0 qid:10032 1:0.130742 2:0.000000 3:0.333333 ... 46:1.000000 #docid=GX140-98-1356607 inc=0.12 prob=0.07013

1 qid:10032 1:0.593640 2:1.000000 3:0.000000 ... 46:0.000000 #docid=GX256-43-0740276 inc=0.01 prob=0.40073

====================================================
```

## How to execute program

1- Download the project from Github

```
git clone https://github.com/LTR4L/ltr4l.git
```

2- go to the project folder

```
$ cd ltr4l
```

3- package jar file

```
$ ant clean package
```

4-run the following command :

```
java -jar LTR4L-X.X.X.jar data/MQ2008/Fold1/train.txt data/MQ2008/Fold1/vali.txt confs/ranknet.config
```

5- open data file:
```
open data.csv
```

## Changing parameters/configurations
Below is an example of a config file:
```
name:RankNet
numIterations:100
learningRate:0.0001
optimizer:sgd
weightInit:normal
reguFunction:L2
reguRate:0.01
layers:10,Sigmoid 1,Identity
```

You must specify the number of nodes and activation for each hidden layer and the output layer.
However, for NNRank, you do not need to specify the final layer.
For example, to add another layer of 3 ReLu nodes and the output layer to Sigmoid, change layers to:
```
layers:10,Sigmoid 3,Relu 1,Sigmoid
```

You can also change the training data and validation data by changing the path of the first and second argument while executing the program:

```
java -jar data/MQ2007/Fold1/train.txt data/MQ2007/Fold1/vali.txt confs/ranknet.config
```




