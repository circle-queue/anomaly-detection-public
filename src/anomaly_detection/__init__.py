import logging
import os

import mlflow

from anomaly_detection import config

mlflow.set_tracking_uri(config.MLFLOW_URI)

logging.getLogger("mlflow.utils.requirements_utils").disabled = True
logging.getLogger("mlflow.system_metrics.system_metrics_monitor").disabled = True

os.environ[
    "BOKEH_CHROMEDRIVER_PATH"
] = r"C:\Users\aes9rsq\.cache\selenium\chromedriver\win64\125.0.6422.14\chromedriver.exe"
