import org.apache.spark.ml.feature._
import org.apache.spark.sql.{DataFrame, Dataset, Row, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.ml.classification.{DecisionTreeClassifier, LogisticRegression, LogisticRegressionModel}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator


object Main extends App {
  val spark: SparkSession = SparkSession
    .builder()
    .master("local[3]")
    .config("spark.executor.extraJavaOptions", "-XX:+UseG1GC")
    .getOrCreate()
  spark.sparkContext.setLogLevel("WARN")

  val df = spark.read.option("header", "false").option("inferSchema", "true").csv("data/example.csv")
  val columns = df.columns.slice(1, df.columns.length)

  val assembler = new VectorAssembler()
    .setInputCols(columns)
    .setOutputCol("features")

   val dfA = assembler.transform(df).select(col(df.columns.head).as("label"), col("features"))

  val dt = new DecisionTreeClassifier().setLabelCol("label")

  dfA.groupBy("label").count().show(false)
  val seed = 1234L
  val trainTest: Array[Dataset[Row]] = dfA.randomSplit(Array(0.7, 0.3), seed)
  val train: Dataset[Row] = trainTest.head
  val test: Dataset[Row] = trainTest.last

  val smote = SMOTE(spark, train, "label", "features")
  val newDf: DataFrame = smote.generate(seed)

  // Fit the models
  val model = dt.fit(train)
  val model2 = dt.fit(newDf)

  val evaluator = new BinaryClassificationEvaluator().setMetricName("areaUnderPR")
  val auRocEvaluator = new BinaryClassificationEvaluator()

  val predictionsWithoutSmote = model.transform(test)
  println(s"AUPRC (before SMOTE) = ${evaluator.evaluate(predictionsWithoutSmote)}")
  println(s"AUROC (before SMOTE) = ${auRocEvaluator.evaluate(predictionsWithoutSmote)}")

  val predictionsWithSmote = model2.transform(test)
  println(s"AUPRC (after SMOTE) = ${evaluator.evaluate(predictionsWithSmote)}")
  println(s"AUROC (after SMOTE) = ${auRocEvaluator.evaluate(predictionsWithSmote)}")
}
