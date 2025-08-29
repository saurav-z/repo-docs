[![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Unlock the power of Amazon SageMaker with the Python SDK, simplifying the entire machine learning workflow from model training to deployment.** Learn more at the [original repository](https://github.com/aws/sagemaker-python-sdk).

## Key Features of the SageMaker Python SDK:

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others.
*   **Built-in Algorithm Support:** Leverage scalable, optimized Amazon algorithms for common machine learning tasks.
*   **Bring Your Own Algorithms:** Easily train and deploy models using custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Flexible Deployment Options:** Deploy models for real-time inference or batch transform.
*   **Integration with Various Frameworks:** Seamlessly integrates with frameworks like Scikit-learn, XGBoost, and more.
*   **Model Monitoring:** Integrated model monitoring capabilities.
*   **Debugging Tools:** Debugging tools available to identify issues with your models.

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

## Core Functionality

*   **[Using the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html)**
*   **[Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)**
*   **[Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)**
*   **[Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)**
*   **[Using Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)**
*   **[Using XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)**
*   **[SageMaker SparkML Serving](https://github.com/aws/sagemaker-python-sdk#sagemaker-sparkml-serving)**

## Additional Features

*   **[SageMaker Automatic Model Tuning](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning)**
*   **[SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)**
*   **[Secure Training and Inference with VPC](https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc)**
*   **[Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)**

## Telemetry

The SDK includes telemetry to help improve the user experience. You can opt-out by configuring the ``TelemetryOptOut`` parameter in the SDK defaults (see the [documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk)).

## AWS Permissions

The SageMaker Python SDK requires the necessary permissions to interact with Amazon SageMaker and AWS services. Review the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/). Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

## Testing

Run unit tests:

```bash
tox tests/unit
```

Run integration tests:

```bash
tox -- tests/integ
```

## Git Hooks

Enable git hooks by running:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Follow the instructions in the [original README](https://github.com/aws/sagemaker-python-sdk) to build and view the documentation.

## SageMaker SparkML Serving

Deploy and perform predictions with SparkML models using MLeap.

**Example:**

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
model_name = 'sparkml-model'
endpoint_name = 'sparkml-endpoint'
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name=endpoint_name)

payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

For more information, please see `SageMaker SparkML Serving Container <https://github.com/aws/sagemaker-sparkml-serving-container>`.