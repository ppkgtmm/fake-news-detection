import os
import logging
import multiprocessing
import pandas as pd
from pyspark.ml.feature import CountVectorizer, IDF
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from modeling.utilities import (
    create_spark_session,
    get_train_test_set,
    evaluate,
    get_feature_name,
)
from hydra.utils import get_original_cwd

log = logging.getLogger(__name__)


def do_tuning(config):
    features = config.variables.feature_var
    target = config.variables.target_var
    out_folder = config.modeling.output_dir

    tune_summary_path = os.path.join(
        get_original_cwd(), out_folder, "lr_tuning_results.csv"
    )
    model_save_path = os.path.join(get_original_cwd(), out_folder, "lr_model")

    spark = create_spark_session(config.spark.app_name, config.spark.driver_memory)

    train, test, text_col_name = get_train_test_set(spark, config)

    vectorized_col_name = get_feature_name(text_col_name, "vectorized")
    vectorizer = CountVectorizer(inputCol=text_col_name, outputCol=vectorized_col_name)
    idf = IDF(inputCol=vectorized_col_name, outputCol=features)
    lr = LogisticRegression(labelCol=target, featuresCol=features, maxIter=10)
    pipeline = Pipeline(stages=[vectorizer, idf, lr])

    param_grid = (
        ParamGridBuilder()
        .addGrid(lr.regParam, config.tuning.lr.reg_param)
        .addGrid(vectorizer.minDF, config.tuning.count_vectorizer.min_df)
        .addGrid(idf.minDocFreq, config.tuning.idf.min_doc_freq)
        .build()
    )

    cross_val = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=param_grid,
        evaluator=BinaryClassificationEvaluator(
            labelCol=target, rawPredictionCol="prediction", metricName="areaUnderROC"
        ),
        numFolds=config.tuning.n_folds,
        seed=config.splitting.seed,
        parallelism=multiprocessing.cpu_count(),
    )

    log.info("Starting grid search parameter tuning process")
    cv_model = cross_val.fit(train)
    log.info("End of grid search parameter tuning process")

    cv_model.save(model_save_path)
    log.info("Model saved to {}".format(model_save_path))

    log.info(
        "Tuned Logistic regression test AUC score : {}".format(
            evaluate(cv_model, test, target)
        )
    )

    pd.DataFrame(
        [
            {cv_model.getEvaluator().getMetricName(): metric, **p}
            for p, metric in zip(param_grid, cv_model.avgMetrics)
        ]
    ).to_csv(tune_summary_path, index=False)

    log.info("Grid search summary saved to {}".format(tune_summary_path))
