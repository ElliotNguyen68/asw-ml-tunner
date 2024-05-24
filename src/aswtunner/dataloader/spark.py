from pyspark.sql import DataFrame as SparkDataFrame, functions as F, SparkSession
from datetime import datetime

from aswtunner.base.dataloader import BaseDataLoader


class DataLoaderSparkOOT(BaseDataLoader):
    cutoff_date: datetime
    df: SparkDataFrame
    datetime_col: str
    train_data = None
    validate_data = None
    sample_ratio: float

    def __init__(
        self, cutoff_date: datetime, datetime_col: str, sample_ratio: float = 1.0
    ) -> None:
        super().__init__()
        self.cutoff_date = cutoff_date
        self.datetime_col = datetime_col
        self.sample_ratio = sample_ratio

    def fit(self, df: SparkDataFrame):
        self.df = df
        self.is_fitted = True

    def get_train_data(self):
        if self.train_data is not None:
            return self.sample(self.train_data, self.sample_ratio)

        super().get_train_data()
        df_train = self.df.filter(F.col(self.datetime_col) < self.cutoff_date)
        df_train_transformed = self.transform(df_train)

        self.train_data = df_train_transformed
        return self.sample(df_train_transformed, self.sample_ratio)

    def get_validate_data(self):
        if self.validate_data is not None:
            return self.sample_ratio(self.validate_data, self.sample_ratio)
        super().get_validate_data()
        df_eval = self.df.filter(F.col(self.datetime_col) >= self.cutoff_date)
        df_eval_transformed = self.transform(df_eval)
        self.validate_data = df_eval_transformed
        return self.sample_ratio(df_eval_transformed, self.sample_ratio)

    def sample(self, df: SparkDataFrame, sample_ratio: float):
        return df.sample(sample_ratio)


class DataLoaderSparkRecommendOOT(DataLoaderSparkOOT):
    user_identity: str
    target: str
    user_common: SparkDataFrame

    def __init__(
        self,
        cutoff_date: datetime,
        datetime_col: str,
        user_identity: str,
        target: str,
        groundtruth_col: str,
        sample_ratio: float = 1.0,
    ) -> None:
        super().__init__(cutoff_date, datetime_col)
        self.user_identity = user_identity
        self.target = target
        self.sample_ratio = sample_ratio
        self.groundtruth_col = groundtruth_col

    def get_validate_data(self):
        if self.validate_data is not None:
            return self.validate_data
        df_eval = self.df.filter(F.col(self.datetime_col) >= self.cutoff_date)
        df_eval_return = (
            df_eval.groupBy(self.user_identity)
            .agg(F.collect_set(self.target).alias(self.groundtruth_col))
            .join(self.user_common, on=self.user_identity)
        )
        self.validate_data = df_eval_return
        return df_eval_return

    def get_train_data(self):
        if self.train_data is not None:
            return self.train_data

        df_train = self.df.filter(F.col(self.datetime_col) < self.cutoff_date)
        df_eval = self.df.filter(F.col(self.datetime_col) >= self.cutoff_date)

        df_user_common = (
            df_train.select(self.user_identity)
            .drop_duplicates()
            .join(
                df_eval.select(self.user_identity).drop_duplicates(),
                on=self.user_identity,
            )
            .sample(self.sample_ratio)
        )
        df_user_common_to_pd = df_user_common.toPandas()
        df_user_common_cache = SparkSession.getActiveSession().createDataFrame(
            df_user_common_to_pd
        )
        self.user_common = df_user_common_cache
        train_data = df_train.join(df_user_common_cache, on=self.user_identity)
        train_transformed = self.transform(train_data)
        self.train_data = train_transformed
        return train_transformed
