# SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Supercharge your machine learning workflow with the SageMaker Python SDK, an open-source library for seamless model training and deployment on Amazon SageMaker.**  [View the original repo](https://github.com/aws/sagemaker-python-sdk)

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Easy Model Deployment:** Deploy your trained models with just a few lines of code.
*   **Comprehensive Documentation:** Access detailed documentation, including API references, for streamlined development.
*   **SparkML Serving:** Deploy and perform predictions against SparkML Models using the MLeap library.
*   **Integration with other SageMaker features:** Utilize with SageMaker features such as Autopilot, Batch Transform, Model Monitoring, Debugger, and Processing.

## Overview

The SageMaker Python SDK simplifies the process of building, training, and deploying machine learning models on Amazon SageMaker.  It provides a high-level, Python-friendly interface for interacting with SageMaker services. Whether you're working with pre-built algorithms, custom code, or popular deep learning frameworks, the SDK empowers you to quickly get your models into production.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version of the SDK using pip:

```bash
pip install sagemaker
```

Or, to install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Environments

*   **Operating Systems:** Unix/Linux and Mac
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12

### Telemetry

The SDK uses telemetry to collect usage data. You can opt out by setting the `TelemetryOptOut` parameter in the SDK defaults configuration. [Learn more](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk)

### AWS Permissions

The SDK generally requires permissions similar to what's needed to use SageMaker. If using an IAM role with a path, grant permission for `iam:GetRole`. [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)

## Testing

### Unit Tests

Run unit tests with tox:

```bash
tox tests/unit
```

### Integration Tests

Run integration tests with tox. Ensure AWS credentials are set and the necessary IAM role is available:

```bash
tox -- tests/integ
```

## Advanced Topics

*   [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
*   [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
*   [SageMaker SparkML Serving](https://github.com/aws/sagemaker-python-sdk#sagemaker-sparkml-serving)
*   [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
*   [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
*   [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)
*   [More Documentation](https://sagemaker.readthedocs.io/en/stable/)

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## SageMaker SparkML Serving

SageMaker SparkML Serving allows you to perform predictions against SparkML models serialized with the MLeap library. Supports Spark 3.3 and MLeap 0.20.0.

Example:

```python
sparkml_model = SparkMLModel(model_data='s3://path/to/model.tar.gz', env={'SAGEMAKER_SPARKML_SCHEMA': schema})
predictor = sparkml_model.deploy(initial_instance_count=1, instance_type='ml.c4.xlarge', endpoint_name='sparkml-endpoint')
payload = 'field_1,field_2,field_3,field_4,field_5'
predictor.predict(payload)
```

[See More on SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)