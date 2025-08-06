<!-- Banner -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK empowers you to effortlessly train and deploy machine learning models on Amazon SageMaker.**  Explore the [original repository](https://github.com/aws/sagemaker-python-sdk) for more details.

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Serving:**  Deploy and perform predictions against SparkML models serialized with the MLeap library.
*   **Model Training and Deployment:** Simplified model training and deployment workflows.
*   **Integration with SageMaker Features:** Seamlessly integrates with SageMaker's features, including automatic model tuning, batch transform, and model monitoring.

## Key Topics:

*   [Installing the SageMaker Python SDK](#installing-the-sagemaker-python-sdk)
*   [Using the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)
*   [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
*   [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
*   [Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
*   [Using XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)
*   [SageMaker Reinforcement Learning Estimators](https://sagemaker.readthedocs.io/en/stable/using_rl.html)
*   [SageMaker SparkML Serving](#sagemaker-sparkml-serving)
*   [Amazon SageMaker Built-in Algorithm Estimators](src/sagemaker/amazon/README.rst)
*   [Using SageMaker AlgorithmEstimators](https://sagemaker.readthedocs.io/en/stable/overview.html#using-sagemaker-algorithmestimators)
*   [Consuming SageMaker Model Packages](https://sagemaker.readthedocs.io/en/stable/overview.html#consuming-sagemaker-model-packages)
*   [BYO Docker Containers with SageMaker Estimators](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-docker-containers-with-sagemaker-estimators)
*   [SageMaker Automatic Model Tuning](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning)
*   [SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)
*   [Secure Training and Inference with VPC](https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc)
*   [BYO Model](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model)
*   [Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)
*   [Amazon SageMaker Operators in Apache Airflow](https://sagemaker.readthedocs.io/en/stable/using_workflow.html)
*   [SageMaker Autopilot](src/sagemaker/automl/README.rst)
*   [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
*   [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
*   [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)

## Installing the SageMaker Python SDK

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems

*   Unix/Linux
*   Mac

### Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The `sagemaker` library collects telemetry data to improve user experience and feature development.  You can opt-out by setting the `TelemetryOptOut` parameter to `true` in your SDK defaults configuration (see the documentation for details).

## AWS Permissions

Amazon SageMaker requires specific permissions to perform operations on your behalf. Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.  If using an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

This SDK is licensed under the Apache 2.0 License.  Copyright Amazon.com, Inc. or its affiliates.  See the [license](http://aws.amazon.com/apache2.0/).

## Running Tests

Install testing dependencies:

```bash
pip install --upgrade .[test]
```

### Unit Tests

Run unit tests using `tox`:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites:

1.  AWS account credentials configured.
2.  IAM role named `SageMakerRole` with the `AmazonSageMakerFullAccess` policy and permissions for Elastic Inference.
3.  Dummy ECR repo created.

Run integration tests:

```bash
# Run specific tests
tox -- -k 'test_i_care_about'

# Run all integration tests (may take a while)
tox -- tests/integ

# Run integration tests in parallel
tox -- -n auto tests/integ
```

## Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable individual hooks by moving them from `.githooks/` to `.git/hooks/`.

## Building Sphinx Docs

Setup a Python environment and install dependencies (see `doc/requirements.txt`). Build and view the docs:

```bash
# Create and activate a conda environment
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# Install dependencies
pip install -r doc/requirements.txt

# Install local version
pip install --upgrade .

# Build documentation
cd sagemaker-python-sdk/doc
make html

# Preview the site
cd _build/html
python -m http.server 8000
```

## SageMaker SparkML Serving

Deploy and perform predictions against SparkML models serialized with MLeap (requires MLeap version 0.20.0 with Spark 3.3):

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

# Invoke endpoint with CSV payload
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information, see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).