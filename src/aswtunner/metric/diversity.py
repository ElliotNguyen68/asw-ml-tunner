from pyspark.sql import (
    functions as F,
    DataFrame as SparkDataFrame,
    SparkSession,
    Window,
)

from aswtunner.base.metric import BaseSparkRecommendMetric


class CoverageSpark(BaseSparkRecommendMetric):
    def evaluate(self, df_recommend: SparkDataFrame, df_groundtruth: SparkDataFrame):
        df_rec_explode = df_recommend.withColumn(
            "rec_up_to_k", F.slice(self.rec_col, 1, self.k)
        ).withColumn("rec_product", F.explode(F.col("rec_up_to_k")))
        no_product_in_rec = (
            df_rec_explode.select("rec_product").drop_duplicates().count()
        )
        df_train_explode = df_groundtruth.withColumn(
            "truth_product", F.explode(self.groundtruth_col)
        )
        no_product_in_train = (
            df_train_explode.select("truth_product").drop_duplicates().count()
        )
        coverage = no_product_in_rec / no_product_in_train
        return coverage


class DistributionSpark(BaseSparkRecommendMetric):
    def evaluate(self, df_recommend: SparkDataFrame, df_groundtruth: SparkDataFrame):
        no_user = df_recommend.select(self.user_identity).drop_duplicates().count()
        df_rec_explode = df_recommend.withColumn(
            "rec_product", F.explode(F.col(self.rec_col))
        )
        window_get_top = Window().orderBy(F.desc("no_distinct_user_get_this_rec"))
        df_rec_get_top_product = (
            df_rec_explode.groupBy("rec_product")
            .agg(
                F.count_distinct(self.user_identity).alias(
                    "no_distinct_user_get_this_rec"
                )
            )
            .withColumn("rank_top_item", F.row_number().over(window_get_top))
            .filter(F.col("rank_top_item") <= self.k)
            .withColumn(
                "percentage_user_get_this_product",
                F.col("no_distinct_user_get_this_rec") / F.lit(no_user),
            )
        )
        distribution = (
            df_rec_get_top_product.select(
                F.mean("percentage_user_get_this_product").alias("distribution")
            )
            .collect()[0]
            .distribution
        )
        return distribution
