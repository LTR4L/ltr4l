# Overview of Algorithms

## Introduction

LTR4L is a Learning to Rank project for Lucene.
The purpose of this document is to provide a general overview of implemented algorithms.
More specifically, it is meant to provide a "big picture," interpretation, or explanation of each network, as well as some experimental results for comparison purposes.
For specific mathematical details and formulation, please see implementation or original articles.

The three different types of machine learning are supervised learning (training labels provided for all labels),
semi-supervised learning (training labels only provided for the training set), and unsupervised learning (no labels provided).
This project currently only implements supervised learning.

In learning-to-rank, there are three different approaches: pointwise (training involves looking at documents of a query independently),
pairwise (training involves looking at pairs of documents; i.e. which document is more relevant), and listwise (training involves looking
at a list of documents for a given query). The following algorithms have been implemented thus far:

1) PRank (Perceptron Ranking)                             [pointwise]
2) OAP-BPM (Online Aggregate Prank-Bayes Point Machine)   [pointwise]
3) NNRank (Neural Network Ranking)                        [pointwise]
4) RankNet (Ranking Network)                              [pairwise]
5) FRankNet                                               [pairwise/listwise]
6) LambdaRank                                             [pairwise/listwise]
7) SortNet (Sorting Network)                              [pairwise]
8) ListNet                                                [listwise]


## Overview

Document features
Neural Network

## Algorithm Guide

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
Weights and thresholds are updated per document.

Note that in terms of a Neural Network, the structure behind PRank is relatively simple. It can be thought of as
a network with only two layers: the input layer, and the output layer. In the input layer, each node
represents a feature of the document (thus the number of nodes is equal to the number of features of the document). In the output layer, there is one node.
The activation for each node in the network is the identity function, and each node's output is multiplied by a weight,
and that becomes one of the inputs of the output node.

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
for i = 0 until i = N
  pRank = pranks[i]
  if (bernoulli(τ) == 1)
    predictAndUpdate(prank)
  if(prank.weightsChanged)
    this.Weights += prank.Weights / N

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

Experiment:




#### RankNet
Type: Feed-forward neural network 
Approach: Pairwise*
Strengths: Relatively high accuracy. Has been widely adopted to much success.
Weaknesses: Very slow for training through all document paris.
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

#### FRankNet

Type: Feed-forward neural network 
Approach: Pairwise*
Strengths: Faster than ranknet, updates less frequently.
Weaknesses: Very slow for training through all document paris.
Network Input: A single document's features
Network Output: Relevance Score
Weights updated per document pair

FRankNet is a modification of RankNet. 



#### LambdaRank

#### SortNet

#### ListNet






