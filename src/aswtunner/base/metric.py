from abc import ABC
from abc import abstractmethod

from pyspark.sql import DataFrame as SparkDataFrame


class BaseMetric(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def evaluate():
        raise NotImplemented


class BaseSparkRecommendMetric(BaseMetric):
    k: int
    rec_col: str
    user_identity: str
    groundtruth_col: str

    def __init__(
        self, k, user_identity: str, rec_col: str, groundtruth_col: str
    ) -> None:
        super().__init__()
        self.k = k
        self.user_identity = user_identity
        self.rec_col = rec_col
        self.groundtruth_col = groundtruth_col

    @abstractmethod
    def evaluate(df_recommend: SparkDataFrame, df_groundtruth: SparkDataFrame):
        raise NotImplemented
