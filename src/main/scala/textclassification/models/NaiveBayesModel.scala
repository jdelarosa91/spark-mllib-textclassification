package textclassification.models

import org.apache.spark.ml.classification.NaiveBayes
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.sql.DataFrame


object NaiveBayesModel {

  def train(train: DataFrame): PipelineModel = {
    // Configure an ML pipeline, which consists of three stages: tokenizer, hashingTF, and lr.
    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")
    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    val estimator = new NaiveBayes()

    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, hashingTF, estimator))

    // Fit the pipeline to training documents.
    pipeline.fit(train)

  }
}
