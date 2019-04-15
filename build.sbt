name := "SMOTE"

version := "0.1"

scalaVersion := "2.12.8"

val spark = new {
  private val ver = "2.4.0"
  val core      = "org.apache.spark" %% "spark-core"      % ver
  val sql       = "org.apache.spark" %% "spark-sql"       % ver
  val ml        = "org.apache.spark" %% "spark-mllib"     % ver
  val streaming = "org.apache.spark" %% "spark-streaming" % ver
}

libraryDependencies ++= Seq(spark.core, spark.sql, spark.ml)