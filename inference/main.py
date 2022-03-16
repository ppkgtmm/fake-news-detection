from modeling.utilities import create_spark_session, get_feature_name
from preprocessing import clean_text
from hydra import compose, initialize
from pyspark.sql.functions import split
from pyspark.ml.tuning import CrossValidatorModel
import pandas as pd


with initialize(config_path="../config", job_name="inference_app"):
    config = compose(config_name="inference.yaml")
    spark = create_spark_session(config.spark.app_name, config.spark.driver_memory)
    model = CrossValidatorModel.load(config.spark.model_path)


def do_prediction(texts):
    text_column = config.variables.text_var
    data = pd.DataFrame(texts, columns=[text_column])
    data[text_column] = data[text_column].map(clean_text)
    spark_df = spark.createDataFrame(data)
    exploded_column = get_feature_name(text_column, "exploded")
    spark_df = spark_df.withColumn(exploded_column, split(spark_df[text_column], " "))
    pred_df = model.transform(spark_df).toPandas()[
        [text_column] + ["probability", "prediction"]
    ]
    pred_df["label"] = pred_df["prediction"].map(
        lambda x: config.variables.target_map[int(x)]
    )
    return pred_df
