from typing import Any, List
from dataclasses import dataclass, asdict


from lightfm import LightFM
from lightfm.data import Dataset
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession, DataFrame, functions as F
from scipy.sparse import csc_matrix

from aswtunner.base.model import BaseRecommenderModel


class LightFMRecommenderSpark(BaseRecommenderModel):
    fit_parameters = ["epochs", "verbose", "num_threads"]
    model_parameters = [
        "no_components",
        "k",
        "n",
        "learning_schedule",
        "loss",
        "learning_rate",
        "rho",
        "epsilon",
        "item_alpha",
        "user_alpha",
        "max_sampled",
    ]
    interaction_col: str
    recommend_col: str
    k: int = 30

    def __init__(
        self,
        user_identity: str,
        target: str,
        interaction_col: str,
        k: int,
        recommend_col: str,
    ) -> None:
        super().__init__(user_identity, target)
        self.interaction_col = interaction_col
        self.recommend_col = recommend_col
        self.k = k

    def init_dataset(self, data: pd.DataFrame):
        dataset = Dataset()
        dataset.fit(data[self.user_identity].unique(), data[self.target].unique())
        self.dataset = dataset

    def transform_dataset(self, data: pd.DataFrame):
        interact = data[[self.user_identity, self.target, self.interaction_col]]
        (interactions, weights) = self.dataset.build_interactions(list(interact.values))
        return (interactions, weights)

    def _to_pandas(self, df: DataFrame):
        return df.toPandas()

    def fit(self, data: DataFrame, **kwargs):
        data_pd = self._to_pandas(data)
        # print(data_pd)
        self.init_dataset(data=data_pd)
        interaction, weight = self.transform_dataset(data=data_pd)
        self.model.fit(interactions=interaction, sample_weight=weight, **kwargs)

    def recommend_for_user(
        self,
        model: LightFM,
        df_user_to_recommend_mapping_index: DataFrame,
        list_available_item_index: List,
        df_mapping_item_index_pd,
        target: str,
    ):
        list_users = df_user_to_recommend_mapping_index.toPandas().user_index.tolist()

        user_combos = np.repeat(list_users, len(list_available_item_index))
        item_combos = np.tile(list_available_item_index, len(list_users))

        mapping_item_reverse = {
            key: value
            for key, value in zip(
                df_mapping_item_index_pd.item_index,
                df_mapping_item_index_pd[target],
            )
        }

        score = model.predict(user_combos, item_combos, num_threads=15)
        sep_score = np.array_split(score, len(list_users))
        result_scores_product = np.stack(sep_score, axis=0)
        top_index_product = np.argsort(-result_scores_product)
        top_index_product = np.array(list_available_item_index)[
            top_index_product[:, : self.k]
        ]

        df_rec_product = pd.DataFrame(
            {
                "user_index": list_users,
                "product_index_dataset": top_index_product.tolist(),
            }
        )
        df_rec_product = df_rec_product.assign(
            product_key_recommendations=lambda x: x.product_index_dataset.apply(
                lambda y: [mapping_item_reverse[z] for z in y]
            )
        )

        df_rec_product_spark = SparkSession.getActiveSession().createDataFrame(
            df_rec_product
        )
        df_rec_product_spark = df_rec_product_spark.join(
            df_user_to_recommend_mapping_index, on="user_index"
        )
        df_rec_product_spark = df_rec_product_spark.withColumnRenamed(
            "product_key_recommendations", self.recommend_col
        )
        return df_rec_product_spark

    def prepare_recommend(
        self,
        data: pd.DataFrame,
    ):
        mapping_product = self.dataset.mapping()[2]
        mapping_user = self.dataset.mapping()[0]
        df_mapping_user_index = pd.DataFrame(
            {
                "user_index": mapping_user.values(),
                "contact_key": mapping_user.keys(),
            }
        )
        df_mapping_item_index = pd.DataFrame(
            {
                "item_index": mapping_product.values(),
                self.target: mapping_product.keys(),
            }
        )
        df_mapping_user_index_spark = SparkSession.getActiveSession().createDataFrame(
            df_mapping_user_index
        )

        df_user_in_eval = SparkSession.getActiveSession().createDataFrame(
            data[[self.user_identity]].drop_duplicates()
        )

        df_user_in_eval_common_train = df_mapping_user_index_spark.join(
            df_user_in_eval, on="contact_key"
        )

        list_product_key_available_to_recommend = (
            df_mapping_item_index.item_index.tolist()
        )
        user_for_eval = df_user_in_eval_common_train
        return (
            user_for_eval,
            list_product_key_available_to_recommend,
            df_mapping_item_index,
        )

    def predict(self, data: DataFrame):
        data_pd = self._to_pandas(data)
        (
            user_for_eval,
            list_product_key_available_to_recommend,
            df_mapping_item_index,
        ) = self.prepare_recommend(data=data_pd)
        recommend = self.recommend_for_user(
            model=self.model,
            df_user_to_recommend_mapping_index=user_for_eval,
            list_available_item_index=list_product_key_available_to_recommend,
            df_mapping_item_index_pd=df_mapping_item_index,
            target=self.target,
        ).select(self.user_identity, self.recommend_col)
        return recommend

    def __call__(self, **kwargs: Any) -> Any:
        self.model = LightFM(**kwargs)


class LightFMRecommenderFeatureSpark(LightFMRecommenderSpark):
    user_feature: pd.DataFrame = None
    item_feature: pd.DataFrame = None
    user_unique_feature_values: List[str] = None
    item_unique_feature_values: List[str] = None

    def _get_unique_value_from_col_array(self, df: DataFrame, column: str):
        df_ = df.withColumn("flat_val", F.explode(column))
        unique_values = (
            df_.select("flat_val").drop_duplicates().toPandas().flat_val.tolist()
        )
        return unique_values

    def set_user_feature(
        self,
        df_user_feature: DataFrame,
        feature_column: str,
    ):
        self.__set_feature(
            df_feature=df_user_feature, feature_column=feature_column, target="user"
        )

    def set_item_feature(
        self,
        df_item_feature: DataFrame,
        feature_column: str,
    ):
        self.__set_feature(
            df_feature=df_item_feature, feature_column=feature_column, target="item"
        )

    def __set_feature(self, df_feature: DataFrame, feature_column: str, target: str):
        unique_features = self._get_unique_value_from_col_array(
            df=df_feature, column=feature_column
        )
        df_feature_pd = df_feature.toPandas()
        if target == "user":
            key = self.user_identity
        else:
            key = self.target
        feature_lightfm = list(
            zip(
                df_feature_pd[key],
                df_feature_pd[feature_column]
                .apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else [])
                .values.tolist(),
            )
        )
        if target == "user":
            self.user_feature = feature_lightfm
            self.user_unique_feature_values = unique_features
        elif target == "item":
            self.item_feature = feature_lightfm
            self.item_unique_feature_values = unique_features

    def init_dataset(self, data: pd.DataFrame):
        assert (
            self.item_unique_feature_values is not None
            and self.user_unique_feature_values is not None
        )
        dataset = Dataset()
        dataset.fit(
            [user_feature[0] for user_feature in self.user_feature],
            [item_feature[0] for item_feature in self.item_feature],
            self.user_unique_feature_values,
            self.item_unique_feature_values,
        )
        self.dataset = dataset

    def fit(self, data: DataFrame, **kwargs):
        data_pd = self._to_pandas(data)
        # print(data_pd)
        self.init_dataset(data=data_pd)
        interaction, weight = self.transform_dataset(data=data_pd)
        self.model.fit(
            interactions=interaction,
            sample_weight=weight,
            user_features=self.dataset.build_user_features(self.user_feature),
            item_features=self.dataset.build_item_features(self.item_feature),
            **kwargs
        )
