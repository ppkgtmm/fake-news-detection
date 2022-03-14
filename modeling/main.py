import os
import logging
import multiprocessing
import pandas as pd
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from modeling.utils import (
    create_spark_session,
    get_train_test_set,
    get_pipeline,
    evaluate,
    get_count_vectorizer,
    get_feature_name,
)
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.pipeline import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

log = logging.getLogger(__name__)


def do_modeling(config):
    features = config.variables.feature_var
    target = config.variables.target_var

    spark = create_spark_session(config.spark.app_name)

    train, test, text_col_name = get_train_test_set(spark, config)

    lr = LogisticRegression(labelCol=target, featuresCol=features, maxIter=10)
    lr_pipeline = get_pipeline(text_col_name, lr, features)

    model_lr = lr_pipeline.fit(train)
    log.info(
        "Logistic regression test AUC score : {}".format(
            evaluate(model_lr, test, target)
        )
    )

    nb = NaiveBayes(labelCol=target, featuresCol=features)
    nb_pipeline = get_pipeline(text_col_name, nb, features)

    model_nb = nb_pipeline.fit(train)
    log.info(
        "Multinomial NB test AUC score : {}".format(evaluate(model_nb, test, target))
    )


def do_tuning(config, out_folder):
    features = config.variables.feature_var
    target = config.variables.target_var
    tune_summary_path = os.path.join(out_folder, "lr_tuning_results.csv")
    model_save_path = os.path.join(out_folder, "lr_model")

    spark = create_spark_session(config.spark.app_name)

    train, test, text_col_name = get_train_test_set(spark, config)

    vectorizers = get_count_vectorizer(text_col_name)
    assembler = VectorAssembler(inputCols=list(vectorizers.keys()), outputCol=features)

    lr = LogisticRegression(labelCol=target, featuresCol=features, maxIter=10)
    pipeline = Pipeline(stages=list(vectorizers.values()) + [assembler, lr])

    param_grid = ParamGridBuilder().addGrid(lr.regParam, config.tuning.lr.reg_param)

    for col_name, values in config.tuning.count_vectorizer.min_df.values():
        vectorizer_col = get_feature_name(col_name, "vectorized")
        vectorizer = vectorizers.get(vectorizer_col, None)
        if vectorizer:
            param_grid.addGrid(vectorizer.minDF, values)

    param_grid = param_grid.build()

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
