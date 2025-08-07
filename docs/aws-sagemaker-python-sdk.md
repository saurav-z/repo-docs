<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Easily build, train, and deploy machine learning models on Amazon SageMaker with the open-source SageMaker Python SDK.**  [View the original repo on GitHub](https://github.com/aws/sagemaker-python-sdk).

<!-- Badges -->
[![PyPI Version](https://img.shields.io/pypi/v/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Conda-Forge Version](https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg)](https://anaconda.org/conda-forge/sagemaker-python-sdk)
[![Python Versions](https://img.shields.io/pypi/pyversions/sagemaker.svg)](https://pypi.python.org/pypi/sagemaker)
[![Code Style: Black](https://img.shields.io/badge/code_style-black-000000.svg)](https://github.com/python/black)
[![Documentation Status](https://readthedocs.org/projects/sagemaker/badge/?version=stable)](https://sagemaker.readthedocs.io/en/stable/)
[![CI Health](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg)](https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml)

## Key Features

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for core ML tasks.
*   **Bring Your Own Algorithms:** Seamlessly train and deploy models from your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Broad Framework Support:** Supports frameworks like Apache MXNet, TensorFlow, PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Flexible Deployment Options:** Deploy models using various instance types and configurations.
*   **SparkML Serving:** Serve predictions against SparkML models serialized with the MLeap library.
*   **Integration with Model Monitoring:**  Easily implement model monitoring solutions for your ML models using SageMaker.
*   **Model Debugging:**  The SDK integrates with SageMaker Debugger for debugging machine learning models.

## Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Python Versions

The SageMaker Python SDK supports:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Telemetry

The `sagemaker` library has telemetry enabled to help us better understand user needs, diagnose issues, and deliver new features. This telemetry tracks the usage of various SageMaker functions.

If you prefer to opt out of telemetry, you can easily do so by setting the ``TelemetryOptOut`` parameter to ``true`` in the SDK defaults configuration. For detailed instructions, please visit `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`__.

## AWS Permissions

Amazon SageMaker can perform only operations that the user permits. You can read more about which permissions are necessary in the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`__.
The SageMaker Python SDK should not require any additional permissions aside from what is required for using SageMaker. However, if you are using an IAM role with a path in it, you should grant permission for ``iam:GetRole``.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. It is copyright Amazon.com, Inc. or its affiliates. All Rights Reserved. The license is available at: <http://aws.amazon.com/apache2.0/>

## Running Tests

For detailed instructions, please refer to the original [README on Github](https://github.com/aws/sagemaker-python-sdk).

## Building Sphinx Docs

For detailed instructions, please refer to the original [README on Github](https://github.com/aws/sagemaker-python-sdk).

## SageMaker SparkML Serving

With SageMaker SparkML Serving, you can now perform predictions against a SparkML Model in SageMaker.
In order to host a SparkML model in SageMaker, it should be serialized with ``MLeap`` library.

For more information on MLeap, see https://github.com/combust/mleap .

Supported major version of Spark: 3.3 (MLeap version - 0.20.0)

Here is an example on how to create an instance of  ``SparkMLModel`` class and use ``deploy()`` method to create an
endpoint which can be used to perform prediction against your trained SparkML Model.

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)
```

Once the model is deployed, we can invoke the endpoint with a ``CSV`` payload like this:

```python
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information about the different ``content-type`` and ``Accept`` formats as well as the structure of the
``schema`` that SageMaker SparkML Serving recognizes, please see `SageMaker SparkML Serving Container <https://github.com/aws/sagemaker-sparkml-serving-container>`_.