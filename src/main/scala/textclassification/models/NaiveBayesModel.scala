package textclassification.models

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature._
import org.apache.spark.sql.DataFrame


object NaiveBayesModel {

  def train(train: DataFrame): PipelineModel = {

    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val cvModel = new CountVectorizer()
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("vectors")
      .setMinDF(1)

    val idf = new IDF()
      .setInputCol(cvModel.getOutputCol)
      .setOutputCol("features")
    val estimator = new NaiveBayes()
      .setModelType("multinomial")

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer,cvModel, idf,  estimator))

    // Fit the pipeline to training documents.
    pipeline.fit(train)

  }
}
