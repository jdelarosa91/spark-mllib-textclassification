package textclassification

import org.apache.spark.sql.SparkSession
import textclassification.models.NaiveBayesModel
import textclassification.utils.{EvaluationMetrics, ProcessingFile}

object Main{

  def main(args: Array[String]): Unit = {

    val t1 = System.nanoTime

    val spark: SparkSession = SparkSession
      .builder()
      .master("local[*]")
      .appName("Text-Classification")
      .getOrCreate()


    val data = ProcessingFile.open(spark, args(0)).cache()

    val (train, test) = ProcessingFile.splitData(data,0.8)

    val model = NaiveBayesModel.train(train)
    EvaluationMetrics.show(model, test)
    val duration = (System.nanoTime - t1) / 1e9d
    println("Seconds: " + duration)
  }
}