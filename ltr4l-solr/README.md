# LTR4L-solr

This document provides how to build, setup, use nlp4l-solr component.

## How to build

This component requires ltr4l classes when building. Then, following is the whole sequence to build.

```
$ git checkout https://github.com/LTR4L/ltr4l
$ cd ltr4l
$ ant clean ivy-bootstrap compile jar
$ cd ltr4l-solr
$ ant clean jar
```

## How to deploy the project jar file

Copy LTR4L_solr jar file to the lib directory of solr-webapp. It usually exists under ${solr_install_dir}/server/solr-webapp/webapp/WEB-INF/lib.

```
$ cp LTR4L_solr-VERSION.jar ${solr_install_dir}/server/solr-webapp/webapp/WEB-INF/lib
```

## How to set up FeaturesRequestHandler

FeaturesRequestHandler is used for extracting features with the features of Apache Lucene. This needs to be registered in solrconfig.xml as following.

```
<requestHandler name="/features" class="org.ltr4l.lucene.solr.server.FeaturesRequestHandler" startup="lazy">
  <lst name="defaults">
    <str name="conf">ltr_features.conf</str>
  </lst>
</requestHandler>
```

Where, ltr_features.conf should be provided in the directory where solrconfig.xml exists and the structure of the file looks like below:

```
{
  "features": [
    {
      "name": "TF in title",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory",
      "params": { "field": "title" }
    },
    {
      "name": "TF in body",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory",
      "params": { "field": "body" }
    },
    {
      "name": "IDF in title",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory",
      "params": { "field": "title" }
    },
    {
      "name": "IDF in body",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory",
      "params": { "field": "body" }
    },
    {
      "name": "TF*IDF in title",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureTFIDFExtractorFactory",
      "params": { "field": "title" }
    },
    {
      "name": "TF*IDF in body",
      "class": "org.ltr4l.lucene.solr.server.FieldFeatureTFIDFExtractorFactory",
      "params": { "field": "body" }
    }
  ]
}
```

+ name : This is feature id, so you can name this parameter freely, but duplication is not allowed
+ class : This is the class used in feature extract, you can choose following now.
  - org.ltr4l.lucene.solr.server.FieldFeatureTFExtractorFactory : For TF feature
  - org.ltr4l.lucene.solr.server.FieldFeatureIDFExtractorFactory : For IDF feature
  - org.ltr4l.lucene.solr.server.FieldFeatureTFIDFExtractorFactory : For TF/IDF feature
+ params : This is the solr field that feature extract will be done.

## How to get feature extracted file(training data)

LTR4L(not LTR4L_solr) contains the command line app to create training data from impresssion log file.

The structure of impression log file is as follows.

```
{
 "data": [
   {
     "query": "query1",
     "impressions": [ "docA", "docB", "docC", "docD", "docE" ],
     "clicks": [ "docA", "docC" ]
   },
   {
     "query": "query1",
     "impressions": [ "docA", "docB", "docC", "docD", "docE" ],
     "clicks": [ "docA" ]
   },
   {
     "query": "query2",
     "impressions": [ "docA", "docB", "docC", "docD", "docE" ],
     "clicks": [ "docD", "docC", "docD" ]
   }
 ]
}
```

+ query : This is query keywords requested to Solr.
+ impressions : These are the document ids returned by the query
+ clicks : These are the clicked(chosen) ids from impressions

With impression log, you can get training data as following command.
```
$ java -cp LTR4L-VERSION.jar org.ltr4l.cli.FeatureExtract [solrUrl] [ltrFeaturesFilename] [impressionLogFile] [trainedDataName(outputFile)] [borders]
```

Borders parameter is the array of float values. This is used for classifying each document by click rate which calculated from impressions and clicks.

example command)
```
$ java -cp LTR4L-VERSION.jar:lib/* org.ltr4l.cli.FeatureExtract http://localhost:8983/solr/techproducts/features ltr_features.conf impressionLog.json training_data.json 1.0,2.4,5.0
```

Then, you can get training data.

## How to make training model

See LTR4L's "How to Execute Training Program" section

## How to deploy the project jar file

Copy model file into Solr's conf directory

```
$ cp ModelFile ${solr_install_dir}/server/solr/${core_name}/conf/.
```


## How to set up QParserPlugin

Add following lines in solrconfig.xml

```
<queryParser name="nn" class="org.ltr4l.lucene.solr.server.DefaultLTRQParserPlugin">
    <lst name="settings">
        <str name="features">features_file_name</str>
        <str name="model">model_file_name</str>
    </lst>
</queryParser>
```

## How to execute rerank using LTR model

```
http://localhost:8983/solr/techproducts/select?indent=on&q=your_query&rq={!rerank reRankQuery=$rqq}&rqq={!nn}your_query&debugQuery=on
```