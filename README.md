# LTR4L

Learning-to-Rank for Apache Lucene (compatibility with Apache Lucene is still a work-in-progress).

The purpose of this document is to provide instructions on how to execute the program, some experimental results.
A general overview is also provided at the end of the document for those who would like to review or learn more about the algorithms.
The general overview is meant to provide a "big picture," interpretation, or explanation of each algorithm, as well as pseudo code.
For specific mathematical details and formulation, please see implementation or original articles.

## Table of Contents

1. Introduction to LTR
2. Execution Instructions
    - Requirements
    - Data
    - How to Execute Program
    - Changing Parameters
    
3. Experiments
4. Overview of Algorithms
    - PRank
    - OAP-BPM
    - NNRank
    - RankNet
    - FRankNet
    - LambdaRank
    - SortNet
    - ListNet
    
## Introduction
Ranking models are an essential component of information retrieval systems.  

Consider the pool of search results for a given search. It is largely impossible to have a pool of search results which will exactly match the set of results the user is looking for. Thus, by vastly increasing the number of search results, one can hope to include all of the desired results of the user. The problem with this, however, is that it becomes much harder for the user to find the desired results among the large pool of search results.  
Ranking serves as means of mitigating this problem. If all of the user's desired results are pushed to the top of the list, then it does not take much effort for the user to find their desired results.  
The ability for a "ranker" to rank the search results accurately is dependent on the ranking model. Learning to Rank is the application of machine learning to construct the ranking model.

There are three different types of machine learning algorithms: supervised learning (labels, or the "correct answer," is provided for all data), unsupervised learning (no labels provided), and reinforcement learning (training through trial and error in a specific environment).

In Learning to Rank, there are three different approaches: pointwise, pairwise, and listwise. In pointwise, training involves looking at the data or documents independently. Each document will have a relevance score, which is unrelated or unaffected by other documents. Because of this, the final rank of the document is not visible to the loss function. In addition, this method does not take into account the fact that documents may actually be related, or that documents are related to a certain query.

The pairwise approach involves considering documents in pairs. Learning occurs by looking at which document is more relevant. For example, the model may make a prediction about which document in a pair is more relevant, and compare that to the labels of each document. Then, the model is adjusted. Depending on the algorithm, the model may output a score for a single document, however the loss function (and thus training) will usually require a pair of documents in this case.

The listwise approach involves looking at lists of documents associated with a query. While the model may output a score for one document, the loss function requires a full list of documents in a query. Thus, during learning, information about all of the documents is necessary.

This project currently only implements algorithms of the supervised learning variety, and thus we will focus on supervised learning in this document. The implemented algorithms include pointwise, pairwise, and listwise approaches.

## Execution Instructions

#### Requirements

You will need to install the following:

1. Java 8+

2. Apache Ant 1.8+ (Not required to run, however the following instructions will assume Apache Ant has been installed.)

#### Data

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

#### How to Execute Program

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

5- open the report file:

```
open report.csv
```

#### Changing Parameters
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

You can also change the gradient descent optimization algorithm used during training.
The following optimization algorithms have been implemented (specification in the config file is case-insensitive):  
SGD  
Adam  
Momentum  
Nesterov  
Adagrad  
RMSProp  
Adamax  
Nadam  
AMSGrad  

The training data and validation data used can be changed by changing the path of the first and second argument while executing the program:

```
java -jar data/MQ2007/Fold1/train.txt data/MQ2007/Fold1/vali.txt confs/ranknet.config
```


## Experiments

In this section, first we provide barcharts which show NDCG@10 for each algorithm, across different datasets.
Then we provide some graphs of NDCG and Loss, as well as the parameters used and elapsed time.
Note that elapsed time can vary quite a bit depending on the parameters (especially on the number of hidden layers, activation, etc...),
and is only provided to give a general idea of the speed of the algorithm. Even with no change in parameters, the elapsed time can vary.
Note that this is the time required for all epochs to finish: i.e. for all training and validation to complete.

#### NDCG Comparisons
Note that SortNet is missing from MQ2007 and OHSUMED, due to a bug.
When it is fixed, the graphs will be updated.

![Alt Text](figures/2007NDCG.jpg)

![Alt Text](figures/2008NDCG.jpg)

![Alt Text](figures/ohsumedNDCG.jpg)


#### PRank

|Parameter|Value|
|:-:|:-:|
|Algorithm|PRank|
|Dataset|LETOR:MQ2007 Fold 1|
|Weights Initialization|Zero|
|Bias Initialization|Zero, Infinity|
|Loss Function|Square Error|
|Epochs|100|
|Time Elapsed|8.7 s|

![Alt Text](figures/PRankNDCG.jpg)
![Alt Text](figures/PRankError.jpg)

#### OAP-BPM
Note: N is the number of PRanks used.

|Parameter|Value|
|:-:|:-:|
|Algorithm|OAP-BPM|
|Dataset|LETOR:MQ2008 Fold 1|
|Weights Initialization|Zero|
|Bias Initialization|Zero, Infinity|
|N|100|
|Bernoulli|0.03|
|Loss Function|Square Error|
|Epochs|100|
|Time Elapsed|10.8 s|

![Alt Text](figures/oapNDCG.jpg)
![Alt Text](figures/oapError.jpg)

#### NNRank

|Parameter|Value|
|:-:|:-:|
|Algorithm|NNRank|
|Dataset|LETOR:MQ2007 Fold 1|
|Optimizer|Momentum|
|Weights Initialization|Gaussian|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 15, 3]|
|Hidden Activation|Identity|
|Output Activation|Sigmoid|
|Loss Function|Square Error|
|Epochs|100|
|Learning Rate|0.01|
|Regularization|L1|
|Regularization Rate|0.01|
|Time Elapsed| s|

work in progress.

#### RankNet

Note: The document pairs used are only the pairs which have different labels (i.e. two documents with label of "0")
are ignored during training. In addition, only a maximum 1/6 of the queries are actually used during training (randomly selected each time).
Note that elapsed time is quite large compared to other algorithms, despite this reduction in the document pairs.
Also note the steady rise in NDCG and fall in loss despite the reduction of document pairs used.

|Parameter|Value|
|:-:|:-:|
|Algorithm|RankNet|
|Dataset|LETOR:MQ2007 Fold 1|
|Optimizer|Adagrad|
|Weights Initialization|Xavier|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 10, 1]|
|Hidden Activation|Identity|
|Output Activation|Sigmoid|
|Loss Function|Cross Entropy|
|Epochs|100|
|Learning Rate|0.00001|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 114.8s|

![Alt Text](figures/RankNetNDCG.jpg)
![Alt Text](figures/RankNetError.jpg)

As an example of the importance of tuning and the difference datasets can make, here are the results of RankNet
performed on MQ2008:

|Parameter|Value|
|:-:|:-:|
|Algorithm|RankNet|
|Dataset|LETOR:MQ2008 Fold 1|
|Optimizer|Adam|
|Weights Initialization|Xavier|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 10, 1]|
|Hidden Activation|Sigmoid|
|Output Activation|Sigmoid|
|Loss Function|Cross Entropy|
|Epochs|100|
|Learning Rate|0.00001|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 37.2s|

![Alt Text](figures/RankNetNDCG2008.jpg)
![Alt Text](figures/RankNetError2008.jpg)


#### FRankNet

Note: FRankNet is still work in progress.

|Parameter|Value|
|:-:|:-:|
|Algorithm|FRankNet|
|Dataset|LETOR:MQ2008 Fold 1|
|Optimizer|Momentum|
|Weights Initialization|Gaussian|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 5, 1]|
|Hidden Activation|Sigmoid|
|Output Activation|Sigmoid|
|Loss Function|Cross Entropy|
|Epochs|100|
|Learning Rate|0.001|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 16.0s|

![Alt Text](figures/FRankNetNDCG2008.jpg)
![Alt Text](figures/FRankNetError2008.jpg)


#### LambdaRank

|Parameter|Value|
|:-:|:-:|
|Algorithm|LambdaRank|
|Dataset|LETOR:MQ2008 Fold 1|
|Optimizer|Adam|
|Weights Initialization|Xavier|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 10, 1]|
|Hidden Activation|Sigmoid|
|Output Activation|Sigmoid|
|Loss Function|Cross Entropy|
|Epochs|100|
|Learning Rate|0.001|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 42.1s|

![Alt Text](figures/LambdaRankNDCG2008.jpg)
![Alt Text](figures/LambdaRankError2008.jpg)


#### SortNet

|Parameter|Value|
|:-:|:-:|
|Algorithm|SortNet|
|Dataset|LETOR:MQ2008 Fold 1|
|Optimizer|Momentum|
|Weights Initialization|Xavier|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 3, 1]|
|Hidden Activation|Sigmoid|
|Output Activation|Sigmoid|
|Loss Function|Square Error|
|Epochs|100|
|Learning Rate|0.01|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 20.8s|


![Alt Text](figures/SortNetNDCG2008.jpg)
![Alt Text](figures/SortNetError2008.jpg)

#### ListNet

|Parameter|Value|
|:-:|:-:|
|Algorithm|ListNet|
|Dataset|LETOR:MQ2007 Fold 1|
|Optimizer|Adam|
|Weights Initialization|Xavier|
|Bias Initialization|Constant (0.1)|
|Layers|[46, 15, 1, 1]|
|Hidden Activation|Identity, Sigmoid|
|Output Activation|Sigmoid|
|Loss Function|Cross Entropy|
|Epochs|100|
|Learning Rate|0.001|
|Regularization|L2|
|Regularization Rate|0.01|
|Time Elapsed| 70.8s|


![Alt Text](figures/ListNetNDCG.jpg)
![Alt Text](figures/ListNetError.jpg)


## Overview of Algorithms

At the end of the overview for each algorithm, pseudo code will be provided.
In general, the flow of training and validation takes the following structure:

```
S = epoch number
for s = 1 to S do:
  train()     //updates weights accordingly
  validate() //calculates NDCG and loss using all documents in the validation set
  report()
end for

```

We will focus on train(), and elaborate on other parts when necessary.
We will also include initialization, and variables.
Note that yi will refer to the label of document xi, unless otherwise stated.

#### PRank
Type: Single Perceptron  
Approach: Pointwise  
Strengths:  Very simple structure, very fast  
Weaknesses: Only linear results/weights can be obtained (not very effective for non-linear data sets).  
Network Input: A single document's features  
Network Output: Relevance score**  
Weights updated by document.  

PRank is essentially the inner product between a weight vector and a document's features. Both vectors are the space of real numbers.
The inner product is the predicted relevance score, and based off of this score, PRank classifies the document.
The document can be classified into categories such as: highly-relevant, relevant, irrelevant or 3-star, 2-star, 1-star, etc...
The simplest classification would be between two classes: relevant and irrelevant.

To determine whether a score falls into a category, PRank looks at "threshold values." Threshold values are similar to a grading system.
For example, if score >= 90, A
if 80 <= score < 90, B
etc...

When PRank is learning, it essentially changes both the weight vector and threshold values, if the predicted score is not correct.
In terms of the grading analogy, it is as if the teacher how many points each problem is worth (weights),
as well as the number of points needed to get a certain score.
The farther away the predicted category is from the actual category, the more the weights and thresholds are changed.
Weights and thresholds are updated per document, and documents are chosen at random.

Note that in terms of a Neural Network, the structure behind PRank is relatively simple. It can be thought of as
a network with only two layers: the input layer, and the output layer. In the input layer, each node
represents a feature of the document (thus the number of nodes is equal to the number of features of the document). In the output layer, there is one node.
The activation for each node in the network is the identity function, and each node's output is multiplied by a weight,
and that becomes one of the inputs of the output node.

Pseudo-code is as follows:
```
S = epoch number
T = Training set of document-label pairs {(x1, y1), (x2, y2),..., (xn, yn))}
V = Validation set of document-label pairs {(x1, y1), (x2, y2),..., (xm, ym))}

Initialization:
weights vector: W = (0, 0, 0, ..., 0)
threshold vector: b = (0, 0, ..., 0) [Note: length of b is # of categories - 1; i.e. last threshold is infinitely large ]
Error function = Square Error

for s=1 to S do:
  shuffle(T) [arrange randomly]
  for each document x(t) in T do:   
    yt' = min(r, r is index of b {1, 2, ..., k}) {(W*x(t) - b[r])  < 0} //start of updateWeights(x)
    if (yt' != yt) then:
      τ[] = new integer array (length = b.length = k) 
      for r = 1 to k - 1:
        integer ytr
        if (yt <= r) then: ytr = -1 else 1 //This is to count mismatched categories.
        if (wx(t) - b[r]) * ytr <=0 then:
          τ[r] = ytr
        else:
          τ[r] = 0
      end for
      τs = Στ[r] for all r
      W = W + τs * X
      b = b - τ
    end if                                 //end of updateWeights(x)
  end for
  validate()
  report()
end for

```

#### OAP-BPM
Type: PRank Network  
Approach: Pointwise  
Strengths: Relatively simple structure, more effective learning than PRank (rise in NDCG, decrease in loss)  
Weaknesses: Slower than PRank, only linear results/weights can be obtained  
Network Input: A single document's features  
Network Output: Relevance score**  
Weights updated by document.  

OAP-BPM is an algorithm which uses N PRanks, and a Bernoulli number τ.  
τ is a number between 0 and 1: 0 < τ <= 1. Bernoulli(τ) returns 1 with a probability of τ, 0 otherwise.  
The average weight of all of the PRanks is equal to the weight of the OAPBPM.  
Each iteration, OAP-BPM will present a document to a particular PRank instance with a probability of τ.  
If a document is presented to a PRank, the PRank will predict the class and update its weights accordingly.  
Then, OAP-BPM's weights are updated by PRank's weight / N.  

Please see below for an example of presentation of documents and update of weights below:  

```
S = epoch number
T = Training set of document-label pairs {(x1, y1), (x2, y2),..., (xn, yn))}
V = Validation set of document-label pairs {(x1, y1), (x2, y2),..., (xm, ym))}
N = number of PRank objects in pRank

Initialization:
weights vector: W = (0, 0, 0, ..., 0)
threshold vector: b = (0, 0, ..., 0) [Note: length of b is # of categories - 1; i.e. last threshold is infinitely large ]
Error function = Square Error

for s = 1 until to S do:
  shuffle(T)
  for each document x(t) in T do:
    for each pR in pRanks do:
      if (bernoulli(τ) == 1) then:
        pR.updateWeights()
        this.weights = this.weights + pr.weights / N
        this.b = this.b + pr.b / N
      end if
    end for
    validate()
    report()
  end for
end for

```

#### NNRank
Type: Feed-forward neural network (Multi-layer perceptron)  
Approach: Pointwise  
Strengths: Relatively fast and accurate, online and batch mode, can handle non-linear data  
Weaknesses:  
Network Input: A single document's features.  
Network Output: Document's classification (for a particular query)  
Weights updated per document.  

NNRank uses a Multi-layer perceptron network, where the output layer has a number of nodes equal to
the number of ordinal categories (for example, if the categories are 1, 2, 3, and 4 stars, there will be 4 output nodes).
The key difference of NNRank (from other implemented MLP algorithms) is that the network directly tries to classify a given document,
rather than output a relevance score.
Strictly speaking, this method falls into the multi-threshold approach.

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}

Initialization:
Error function = square error
network = constructNodes() [2 dimensional list of nodes]
initializeWeights() [use specified random initialization strategy]

for s = 1 until S do:
  for all document x in T do:
    output = predict(x's features)
    label = x.label
    if (output != label) then:
      backProp(x)
      updateWeights(x)
    end if
  end for
  validate()
  report()
end for


predict(features):
  threshold = 0.5
  forwardProp(features)
  for nodeId = 0 until network.finalLayer.length do:
    node = network.finalLayer[nodeId]
    if (node.getOutput < threshold) then:
      return nodeId - 1
    end if
  end for
  return network.finalLayer.length - 1

```


#### RankNet
Type: Feed-forward neural network  
Approach: Pairwise*  
Strengths: Relatively high accuracy. Has been widely adopted to much success.  
Weaknesses: Very slow for training through all document pairs.  
Network Input: A single document's features  
Network Output: Relevance Score  
Weights updated per document pair  

RankNet uses a Multi-Layer perceptron network, where the output layer has one node.
It works to solve the preference learning problem directly, rather than solving an
ordinal regression (i.e. a classification problem). While only a single document is
necessary for forward-propagation (and thus ranking), a pairwise cross-entropy error
function is used to measure loss, and thus two documents are required for backpropagation
(as loss is defined over two documents, the derivative of the loss must be taken with respect
to document pairs). Therefore, RankNet is considered a pairwise approach, document pairs are
used as instances during training.

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}
TP = All pairs of documents (x1, x2) associated with each q in T, such that y1 > y2.
VP = All pairs of documents (x1, x2) associated with each q in V, such that y1 > y2.

Initialization:
Error function = cross entropy
network = constructNodes() //2 dimensional list of nodes
initializeWeights() //use specified random initialization strategy

for s = 1 until S do:
  for all document pairs (x1, x2) in T do:
    threshold = 0.5
    s1 = forwardProp(x1)
    s2 = forwardProp(x2)
    delta = s1 - s2
    if (delta < threshold) then:
      sigma = Sigmoid.output(delta)
      backProp(Sigma)
      forwardProp(x1)
      backProp(-Sigma)
    end if
  end for
  validate()
  report()
end for

```

#### FRankNet

Type: Feed-forward neural network  
Approach: Pairwise, Listwise*  
Strengths: Faster than ranknet, updates less frequently.  
Weaknesses:  
Network Input: A single document's features  
Network Output: Relevance Score  
Weights updated per query  

FRankNet is a modification of RankNet. The structure of the network in FRankNet is unchanged from RankNet.
The difference lies in the way that the weights are updated. RankNet looks at how well the current model predicts
a pair of documents and adjusts the model. FRankNet looks at how well the current model predicts for many different pairs,
then adjusts the model.

In RankNet, first the output function is calculated for a pair of documents in a particular query,
then the derivative of the cost function is taken with respect to the difference of scores of the pair, and that is used to
update the weights. When updating for every pair of documents, two backpropagations are required for every pair.
Thus, for N documents in a query, the total number of updates per query is:  
N(N-1)/2,  
and the total number of times backpropagation occurs is:
N(N-1).

In FRankNet, the weights are updated per query.
The output function is first calculated for every pair in the query and combined (added and subtracted appropriately).
Then, the derivative of the cost function is taken with respect to the score of a document (not the difference of scores of a document pair) using the output.
This derivative is called λi (for document i) and it is used for backpropagation. After this is done for every document in the query, the weights are updated.
Thus, for N documents in a query, the total number of updates per query is:  
`1`, and the total number of times backpropagation occurs is: `N`.

The approach is pairwise because the cost function is defined for document pairs, and listwise because
all of the document pairs (and thus all of the documents) in a query are used; thus a document list for the query
is required.

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}
TP = All pairs of documents (x1, x2) associated with each q in T, such that y1 > y2.
VP = All pairs of documents (x1, x2) associated with each q in V, such that y1 > y2.

Initialization:
Error function = cross entropy
network = constructNodes() //2 dimensional list of nodes
initializeWeights() //use specified random initialization strategy

for s = 1 until S do:
  for all queries q in T do:
    lambdas = new map
    ranks = new map
      for each doc x in q, do:
        ranks.put(x, forwardProp(x)) //this is for speed up
        lambdas.put(x, 0)
      end for
    for each document pair (x1, x2) in q do:
      delta = ranks.get(x2) - ranks.get(x1)
      lambda = Sigmoid.output(delta)
      lambdas[x1] -= lambda
      lambdas[x2] += lambda
    end for
    for each document x1 in q do:
      forwardProp(x1)
      backProp(lambdas[x1])
    end for
  updateWeights()
  validate()
  report()
end for
```


#### LambdaRank
Type: Feed-forward neural network  
Approach: Pairwise, Listwise*  
Strengths: Faster than ranknet, updates less frequently.  
Weaknesses:  
Network Input: A single document's features  
Network Output: Relevance Score  
Weights updated per query  

LambdaRank is a modification of FRankNet. The structure of the network remains unchanged.

The main difference is when calculating the λi by looking at all pairs with document i (explained in FRankNet),
the value is multiplied by |ΔNDCG| of swapping the two documents. Please refer to the original article for the reasoning behind
the use of this term.

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}
TP = All pairs of documents (x1, x2) associated with each q in T, such that y1 > y2.
VP = All pairs of documents (x1, x2) associated with each q in V, such that y1 > y2.

Initialization:
Error function = cross entropy
network = constructNodes() //2 dimensional list of nodes
initializeWeights() //use specified random initialization strategy

for s = 1 until S do:
  for all queries q in T do:
    lambdas = new map
    ranks = new map
      for each doc x in q, do:
        ranks.put(x, forwardProp(x)) //this is for speed up
        lambdas.put(x, 0)
      end for
    for each document pair (x1, x2) in q do:
      calculate ΔNDCG
      delta = ranks.get(x2) - ranks.get(x1)
      lambda = Sigmoid.output(delta)
      lambdas[x1] -= lambda * ΔNDCG
      lambdas[x2] += lambda * ΔNDCG
    end for
    for each document x1 in q do:
      forwardProp(x1)
      backProp(lambdas[x1])
    end for
  updateWeights()
  validate()
  report()
end for
```
 
#### SortNet
Type: Comparative Neural Network  
Approach: Pairwise  
Strengths:   
Weaknesses:  
Network Input: Two document's features
Network Output: Preference between the two documents (which document is more relevant).  
For example, for an input of {x1, x2}, the output will be {o1, o2}.  
if o1 > o2, then o1 is more relevant than o2. If o2 > o1, then o2 is more relevant than o1.
Thus, the targets (given two labels) will be either {1,0} or {0,1}
Weights updated per document (/ pair)

The structure of SortNet's neural network is different from the rest of the networks implemented thus far in this project.
Apart from the fact that the features of two documents are required for forward propagation, the output layer must have two nodes,
and each hidden layer must have an even number of nodes. The reason for this is that there is symmetry in the network, such that
the weights between ni and n1 are equal to ni' and n1', and the weights between ni and n1' are equal to ni' and n1.

Note: As the output of the network is the preference between the two documents, there is no way to directly calculate the score of a document,
nor is there a way to classify the document (for ordinal regression).

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}
TP = All pairs of documents (x1, x2) associated with each q in T, such that y1 > y2.
VP = All pairs of documents (x1, x2) associated with each q in V, such that y1 > y2.

Initialization:
Error function = square error
network = constructNodes() //2 dimensional list of nodes; comparative neural network
initializeWeights() //use specified random initialization strategy
targets[][] = {{1,0},{0,1}}

for s = 1 until S do:
  for all queries q in T do:
  threshold = 0.5
    for all documents x1 in q do:     //This is a variation. It does not look at all pairs!
      get a random document x2 in q such that x2 != x1
      delta = y1 - y2
      if (delta != 0) then:
        prediction = predict(x1, x2)
        if (prediction * delta < threshold) then:
          if (delta > 0) then: backprop (targets[0], errorFunc)
          else: backProp(targets[1], errorFunc)
          updateWeights()
        end if
      end if
    end for
  validate()
  report()
  end for
end for


predict(features)
  output = forwardProp(features)
  return output[0] - output[1]


```

#### ListNet
Type: Feed-forward neural network  
Approach: Listwise 
Strengths:  
Weaknesses:  
Network Input: A single document's features  
Network Output: Relevance Score  
Weights updated per query 

ListNet can be considered a "listwise version" of RankNet. The main difference is that the cross entropy loss is calculated using a list
of documents rather than a pair of documents, and the probability used for the cross entropy loss is top one probability (see original paper)
rather than the probability of one document being more relevant than another.
To see the key differences, please look at implementation of ListNetMLP.backProp and ListNetTrainer.calculateLoss().

```
S = epoch number
T = Training set of queries with associated document-label pairs {q1, q2, ..., qn} (each q has a list of documents w/ labels)
V = Validation set of queries with associated document-label pairs {q1, q2, ..., qm}

Initialization:
Error function = Entropy
network = constructNodes() //2 dimensional list of nodes
initializeWeights() //use specified random initialization strategy
targets[][] = {{1,0},{0,1}}

for s = 1 until S do:
  for all queries q in T do:
    for all documents x in q do:
      forwardProp(x)
      backProp(x)
    end for
    updateWeights()
  end for
  validate()
  report()
end for

```