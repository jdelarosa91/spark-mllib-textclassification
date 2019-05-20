package textclassification

import org.apache.spark.sql.{DataFrame, SparkSession}

object ProcessingFile {

  def open(spark: SparkSession, path: String): DataFrame = {
    spark.read.json(path)
      .select("text", "stars")
  }
}
