![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models

**Supercharge your machine learning workflow with the Amazon SageMaker Python SDK, an open-source library for seamless model training and deployment on Amazon SageMaker.** Access the original repo [here](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy your trained models for real-time inference.
*   **Integration:** Compatible with various services and tools, including Apache Airflow and MLeap.
*   **Comprehensive Documentation:** Access detailed documentation, including an API reference, on [Read the Docs](https://sagemaker.readthedocs.io/).

## Installation

Install the latest version using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Alternatively, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Core Functionality

The SDK provides a comprehensive suite of tools for all stages of the machine learning lifecycle, from data preparation to model monitoring.  Key functionalities include:

*   **Training:**
    *   `MXNet <https://sagemaker.readthedocs.io/en/stable/using_mxnet.html>`__
    *   `TensorFlow <https://sagemaker.readthedocs.io/en/stable/using_tf.html>`__
    *   `PyTorch <https://sagemaker.readthedocs.io/en/stable/using_pytorch.html>`__
    *   `Scikit-learn <https://sagemaker.readthedocs.io/en/stable/using_sklearn.html>`__
    *   `XGBoost <https://sagemaker.readthedocs.io/en/stable/using_xgboost.html>`__
    *   `Amazon SageMaker Built-in Algorithm Estimators <src/sagemaker/amazon/README.rst>`__
    *   `Using SageMaker AlgorithmEstimators <https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators>`__
*   **Deployment:**
    *   `BYO Docker Containers with SageMaker Estimators <https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators>`__
    *   `SageMaker Batch Transform <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform>`__
    *   `Inference Pipelines <https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines>`__
    *   `SageMaker SparkML Serving <#sagemaker-sparkml-serving>`__
*   **Management & Monitoring:**
    *   `SageMaker Automatic Model Tuning <https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning>`__
    *   `Model Monitoring <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html>`__
    *   `SageMaker Debugger <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html>`__
    *   `SageMaker Processing <https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html>`__

## Supported Technologies

*   **Programming Languages:** Python 3.9, 3.10, 3.11, and 3.12
*   **Operating Systems:** Unix/Linux and Mac

## SageMaker SparkML Serving

Host SparkML models in SageMaker using the MLeap library.  Use the SparkMLModel class to deploy models serialized with MLeap and perform predictions against them.

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Then invoke endpoint with a CSV payload.

## Contributing

See instructions on testing and building documentation below.  Contributions are welcome!

## Running Tests

**Unit tests:**

```bash
tox tests/unit
```

**Integration tests:**

Prerequisites: AWS account credentials, IAM role named SageMakerRole with appropriate permissions, and a dummy ECR repo.

```bash
tox -- tests/integ
# OR, run tests in parallel
tox -- -n auto tests/integ
```

## Building Documentation

```bash
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0
pip install -r doc/requirements.txt
pip install --upgrade .
cd sagemaker-python-sdk/doc
make html
cd _build/html
python -m http.server 8000
```

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.