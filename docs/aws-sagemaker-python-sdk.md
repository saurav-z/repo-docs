[![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**The SageMaker Python SDK empowers you to seamlessly train, tune, and deploy your machine learning models on Amazon SageMaker.** Access the [original repository](https://github.com/aws/sagemaker-python-sdk) for more details.

Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like **Apache MXNet**, **TensorFlow**, and **PyTorch**.
*   **Built-in Algorithms:** Leverage scalable **Amazon algorithms** optimized for SageMaker and GPU training.
*   **Bring Your Own Algorithms:** Utilize your custom algorithms built into SageMaker-compatible Docker containers.
*   **SparkML Serving:** Deploy SparkML models using the `MLeap` library and perform predictions.
*   **Integration with SageMaker features:** Support for SageMaker Autopilot, Batch Transform, Debugger, Model Monitoring, Processing, and more.

## Getting Started

### Installation

Install the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Or, to install the latest version from the PyPI, specify the latest version tag:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Alternatively, install from source:

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

## Deep Dive into SageMaker with the SDK

Here's a look at the core functionalities available within the SageMaker Python SDK, enabling comprehensive model lifecycle management:

*   [Installing SageMaker Python SDK](#installing-the-sagemaker-python-sdk)
*   [Using the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)
*   [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
*   [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
*   [Using Chainer](https://sagemaker.readthedocs.io/en/stable/using_chainer.html)
*   [Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
*   [Using Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)
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

## Usage & Examples

See the `Read the Docs <https://sagemaker.readthedocs.io>`_ for detailed documentation and API references.

### SageMaker SparkML Serving

Easily deploy and perform predictions against SparkML Models using the SageMaker Python SDK. Ensure your SparkML model is serialized with the `MLeap` library.

Supported Spark version: 3.3 (MLeap version - 0.20.0)

Example deployment:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Invoke endpoint:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For details, see [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).

## Contributing

Contributions are welcome! Please refer to the [CONTRIBUTING.md](https://github.com/aws/sagemaker-python-sdk/blob/master/CONTRIBUTING.md) for guidelines.

### Telemetry

The `sagemaker` library includes telemetry to help us understand user needs and improve the library. You can opt out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See the [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) documentation for instructions.

### AWS Permissions

Amazon SageMaker performs operations on your behalf. Ensure your IAM role has the necessary permissions as detailed in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).  If using an IAM role with a path, grant permission for `iam:GetRole`.

## Testing

### Unit Tests

Run unit tests with tox:

```bash
tox tests/unit
```

### Integration Tests

Prerequisites:

1.  AWS account credentials are available in the environment.
2.  IAM role named `SageMakerRole` with the `AmazonSageMakerFullAccess` policy.
3.  Create dummy ECR repo to run remote_function tests

Run integration tests selectively:

```bash
tox -- -k 'test_i_care_about'
```

Or run all integration tests:

```bash
tox -- tests/integ
```

Or run integration tests in parallel:

```bash
tox -- -n auto tests/integ
```

## Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

Enable an individual hook:

```bash
mv .githooks/your-hook .git/hooks/
```

## Documentation

### Building Sphinx docs

Follow these instructions to build the Sphinx documentation locally:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt

pip install --upgrade .

cd sagemaker-python-sdk/doc
make html

cd _build/html
python -m http.server 8000
```

Access the generated documentation at http://localhost:8000.