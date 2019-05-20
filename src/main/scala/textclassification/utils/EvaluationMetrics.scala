package textclassification.utils

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.DataFrame

object EvaluationMetrics {

  def show(model: PipelineModel, test: DataFrame): Unit = {
    val predictions = model.transform(test)
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val mAccuracy = evaluator.evaluate(predictions)
    println("Test set accuracy = " + mAccuracy)
  }
}
