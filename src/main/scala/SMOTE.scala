import breeze.linalg.DenseVector
import org.apache.spark.ml.feature.{BucketedRandomProjectionLSH, BucketedRandomProjectionLSHModel}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{asc, col}

//https://arxiv.org/pdf/1106.1813.pdf
class SMOTE(spark: SparkSession,
            data: DataFrame,
            bucketLength: Int,
            numOfHashTables: Int,
            labelCol: String,
            featureCol: String) {
  private val minorityLbl =
    data
      .groupBy(labelCol)
      .count()
      .orderBy(asc("count"))
      .take(1)
      .head.getAs[Int](labelCol)
  private val minorityData = data.filter(col(labelCol) === minorityLbl)
  minorityData.cache()

  private val brp: BucketedRandomProjectionLSH = new BucketedRandomProjectionLSH()
    .setBucketLength(bucketLength)
    .setNumHashTables(numOfHashTables)
    .setInputCol(featureCol)
    .setOutputCol("hashes")
  val model: BucketedRandomProjectionLSHModel = brp.fit(minorityData)

  def generate(seed: Long, k: Int = 5, factor: Int = 1): DataFrame = {
    assert(factor <= k, "Number of nearest neighbours (k) should greater or equal than factor!")
    val r = scala.util.Random
    r.setSeed(seed)

    val syntheticDataRdd = spark.sparkContext.parallelize {
      minorityData.collect().flatMap { row =>
        generate(row, r, k, factor)
      }
    }

    val syntheticData = spark.createDataFrame(syntheticDataRdd, data.schema)
    val ratio = minorityData.count().toDouble / syntheticData.count().toDouble

    data.union(syntheticData.sample(ratio, seed))
  }

  private def generate(sourceRow: Row, r: scala.util.Random, k: Int, factor: Int): Seq[Row] = {
    val key = sourceRow.getAs[org.apache.spark.ml.linalg.Vector](featureCol)
    val key2Search = new DenseVector(key.toArray)

    val neighbors =
      model
        .approxNearestNeighbors(minorityData, key, k)
        .filter(col("distCol") > 0)
        .select(labelCol, featureCol)
        .collect()

    //TODO: check the mathematical correctness of sampling
    neighbors.indices
      .map(ind => ind -> r.nextDouble())
      .sortBy(_._2)(Ordering.Double.reverse)
      .take(factor)
      .map {
        case (ind, _) =>
          val row = neighbors(ind)
          val v = row.getAs[org.apache.spark.ml.linalg.Vector](featureCol)
          val v1 = new DenseVector(v.toArray)
          Row(minorityLbl, Vectors.dense((key2Search + r.nextDouble * (v1 - key2Search)).toArray))
      }
  }
}

object SMOTE {

  def apply(spark: SparkSession,
            data: DataFrame,
            labelCol: String,
            featureCol: String,
            bucketLength: Int = 10,
            numOfHashTables: Int = 5): SMOTE =
    new SMOTE(spark, data, bucketLength, numOfHashTables, labelCol, featureCol)
}