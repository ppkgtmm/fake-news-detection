import os
import streamlit as st
from hydra.utils import get_original_cwd
from pyspark.ml.tuning import CrossValidatorModel
from pyspark.sql.functions import split
from modeling.utilities import create_spark_session, get_feature_name
from preprocessing import clean_text
import pandas as pd
import plotly.express as px


def serve_model(config):
    spark = create_spark_session(config.spark.app_name, config.spark.driver_memory)
    column = config.variables.text_var
    model_full_path = os.path.join(get_original_cwd(), config.spark.model_path)
    model = CrossValidatorModel.load(model_full_path)

    st.subheader("Fake news detection with machine learning")
    text_input = st.text_area("", height=100, max_chars=500)
    button = st.button("Analyze")

    if button and text_input:
        data = (clean_text(text_input),)
        df = spark.createDataFrame([data]).toDF(column)
        new_column = get_feature_name(column, "exploded")
        df = df.withColumn(new_column, split(df[column], " "))
        pred_df = model.transform(df).toPandas()
        plot_data = pd.DataFrame(
            {
                "probability": pred_df["probability"][0],
                "label": config.variables.target_map,
            }
        )
        fig = px.bar(
            plot_data,
            x="probability",
            y="label",
            color="label",
            orientation="h",
            color_discrete_map={
                k: v
                for k, v in zip(
                    config.variables.target_map, config.variables.target_colors
                )
            },
        )
        # fig = px.pie(plot_data, values='probability', names='label', color="label",
        #              color_discrete_map={k: v for k, v in zip(config.variables.target_map, config.variables.target_colors)}
        #              )
        fig.update_layout(title="Fake news prediction result", title_x=0.5, height=400)
        st.write(fig)
