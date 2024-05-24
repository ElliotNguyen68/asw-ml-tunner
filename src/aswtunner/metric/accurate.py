from pyspark.sql import DataFrame as SparkDataFrame, functions as F, Window
from aswtunner.base.metric import BaseSparkRecommendMetric


class MapK(BaseSparkRecommendMetric):
    def evaluate(self, df_recommend: SparkDataFrame, df_groundtruth: SparkDataFrame):
        """
        truth_col:str, list of recommended products
        pred_col:str, list of known positive products
        """

        df_moc_rec = (
            df_recommend.select("*", F.posexplode(self.rec_col))
            .withColumn("up_to_k", F.col("pos") + 1)
            .drop("col", "pos")
        )

        df_rec_up_to_k = df_moc_rec.filter(F.col("up_to_k") <= self.k).withColumn(
            "rec_up_to_k", F.slice(F.col(self.rec_col), F.lit(1), F.col("up_to_k"))
        )

        df_get_hit_up_to_k = (
            df_rec_up_to_k.join(df_groundtruth, on=self.user_identity)
            .withColumn(
                "no_relevance_up_to_k",
                F.size(
                    F.array_intersect(F.col("rec_up_to_k"), F.col(self.groundtruth_col))
                ),
            )
            .filter(F.col("no_relevance_up_to_k") > 0)
        )
        window_get_hit = (
            Window()
            .partitionBy(self.user_identity, "no_relevance_up_to_k")
            .orderBy("up_to_k")
        )
        df_get_hit_up_to_k = df_get_hit_up_to_k.withColumn(
            "order_up_to_k_when_hit", F.row_number().over(window_get_hit)
        ).filter(F.col("order_up_to_k_when_hit") == 1)

        df_precision_up_to_k = df_get_hit_up_to_k.withColumn(
            "precision_up_to_k", F.col("no_relevance_up_to_k") / F.col("up_to_k")
        )
        df_full_common_user = (
            df_recommend.select(self.user_identity)
            .join(df_groundtruth.select(self.user_identity), on=self.user_identity)
            .select(self.user_identity)
            .drop_duplicates()
        )
        # df_precision_up_to_k.show()

        df_apk_user = (
            df_precision_up_to_k.withColumn(
                "len_grouth_truth", F.size(F.col(self.groundtruth_col))
            )
            .groupBy(self.user_identity, "len_grouth_truth", self.groundtruth_col)
            .agg(F.sum(F.col("precision_up_to_k")).alias("apk"))
            .withColumn(
                # 'apk',F.col('apk')/F.lit(k)
                "apk",
                F.col("apk") / F.least(F.lit(self.k), F.col("len_grouth_truth")),
            )
            .join(df_full_common_user, on=self.user_identity, how="right")
            .fillna(0, ["apk"])
        )

        df_apk_user = (
            df_apk_user.select("apk", self.user_identity)
            .join(
                df_groundtruth.select(self.user_identity, self.groundtruth_col),
                on=self.user_identity,
            )
            .join(
                df_recommend.select(self.user_identity, self.rec_col),
                on=self.user_identity,
            )
        )

        df_apk_user = df_apk_user.withColumnRenamed("apk", "apk_out")

        return df_apk_user.select(F.mean("apk_out").alias("mapk")).collect()[0].mapk
