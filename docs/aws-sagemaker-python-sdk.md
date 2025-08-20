# SageMaker Python SDK: Train and Deploy Machine Learning Models

**Unlock the power of Amazon SageMaker with the open-source SageMaker Python SDK, simplifying your machine learning workflow from training to deployment.**  ([Original Repo](https://github.com/aws/sagemaker-python-sdk))

[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

The SageMaker Python SDK is your gateway to building, training, and deploying machine learning models on Amazon SageMaker. This library provides a streamlined experience for data scientists and ML engineers, offering flexibility and power.

## Key Features

*   **Framework Support:** Train models with popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Utilize Amazon's scalable, optimized machine learning algorithms.
*   **Custom Algorithms:** Train and deploy models using your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Integration with SparkML:** Seamlessly deploy and perform predictions with SparkML models using MLeap.
*   **Flexible Training Options:** Supports various training methods including distributed training and hyperparameter tuning.
*   **Model Monitoring and Debugging:** Includes features for model monitoring, and debugging.
*   **Comprehensive Documentation:** Extensive documentation with API reference for easy understanding and implementation.

## Getting Started

### Installation

Install the latest version using pip:

```bash
pip install sagemaker
```

Or install from source:

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

The SDK includes telemetry to help improve user experience. You can opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See the [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for details.

## AWS Permissions

The SDK requires the same permissions as Amazon SageMaker. See the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.  If you're using an IAM role with a path, grant permission for `iam:GetRole`.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## Testing

Run tests using tox, install testing dependencies with:
```bash
pip install --upgrade .[test]
```

Then, run the unit tests with:

```bash
tox tests/unit
```

And the integration tests with:

```bash
tox -- tests/integ
```

(Additional steps may be required to run integration tests, see original documentation.)

## Build Documentation

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

# View in browser
cd _build/html
python -m http.server 8000
```

## SageMaker SparkML Serving

Deploy SparkML models with MLeap using the `SparkMLModel` class.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

See [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) for details.