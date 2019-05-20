package textclassification.utils

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col

object ProcessingFile {

  val cleanText = (text: String) => {
    text.replaceAll("[(?¿!¡)$/#*:,.\\\"|]", "")
  }

  def open(spark: SparkSession, path: String): DataFrame = {

    val cleanTextUDF = spark.udf.register("cleanText", cleanText)
    spark.read.json(path)
      .select("text", "stars")
      .withColumn("text", cleanTextUDF(col("text")))
      .withColumnRenamed("stars", "label")
  }

  def splitData(data: DataFrame, i: Double): (DataFrame, DataFrame) = {
    val p = if (i >= 1 || i <= 0) {
      0.9
    }
    else {
      i
    }

    val arr = data.randomSplit(Array(p, 1-p), seed = 12345)
    (arr(0), arr(1))
  }


}
