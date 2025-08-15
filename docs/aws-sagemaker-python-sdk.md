[![SageMaker](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**Unlock the power of Amazon SageMaker with the open-source Python SDK, enabling you to build, train, and deploy machine learning models seamlessly.** This SDK provides a simplified interface for interacting with SageMaker, making it easier than ever to bring your ML projects to life.  Learn more at the [original repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithm Support:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using custom algorithms packaged in SageMaker compatible Docker containers.
*   **Framework Compatibility:** Seamlessly integrate with Chainer, PyTorch, Scikit-learn, and XGBoost.
*   **Model Deployment:** Easily deploy your trained models for real-time inference.
*   **SparkML Integration:** Deploy and perform predictions against SparkML models serialized with the MLeap library.
*   **Comprehensive Documentation:** Access detailed documentation, including API references, on [Read the Docs](https://sagemaker.readthedocs.io/).

## Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```
*   The latest version can be found on [PyPI](https://pypi.org/project/sagemaker/)

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Technologies and Features

*   **Supported Operating Systems:** Unix/Linux, Mac.
*   **Supported Python Versions:**
    *   3.9
    *   3.10
    *   3.11
    *   3.12
*   **Telemetry:** The SDK includes telemetry to improve the product.  Opt-out instructions are available in the documentation.
*   **AWS Permissions:** Learn about necessary permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Example: Deploying a SparkML Model

Here's how to deploy a SparkML model:

```python
from sagemaker.sparkml.model import SparkMLModel

sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge')
```

For more information on model deployment, please see the [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container).

## Testing

Run tests with tox:

```bash
tox tests/unit
```

Or run integration tests:

```bash
tox -- tests/integ
```

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).