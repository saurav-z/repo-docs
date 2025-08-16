# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**Easily build, train, and deploy machine learning models on Amazon SageMaker with the open-source SageMaker Python SDK, streamlining your ML workflow.**  [View the original repo](https://github.com/aws/sagemaker-python-sdk)

![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, pre-built Amazon algorithms optimized for SageMaker and GPU training.
*   **Bring Your Own Algorithms:**  Seamlessly integrate your custom algorithms built into SageMaker-compatible Docker containers.
*   **SparkML Serving:**  Deploy SparkML models using the MLeap library.
*   **Extensive Documentation:** Comprehensive documentation and API reference available on [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).
*   **Flexible Deployment:** Deploy models with a variety of instance types and configurations.
*   **Model Monitoring:** Monitor deployed models.
*   **Model Debugging:** Use the SageMaker Debugger.
*   **Model Autotuning:** Automatic model tuning.
*   **Batch Transform:** Support for batch transform jobs.

## Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

You can also install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Features

*   **Operating Systems:** Unix/Linux and Mac.
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12.

## AWS Permissions

Ensure your AWS Identity and Access Management (IAM) role has the necessary permissions as detailed in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Telemetry

The SDK includes telemetry to improve user experience. Opt-out by setting the `TelemetryOptOut` parameter to `true` in your SDK configuration. For detailed instructions, please visit `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>`__.

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## Resources
*   [Read the Docs](https://sagemaker.readthedocs.io/en/stable/)
*   [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)

## Testing

### Unit Tests
```bash
tox tests/unit
```

### Integration Tests

**Prerequisites:**

1.  AWS account credentials in the environment.
2.  An IAM role named `SageMakerRole` with the `AmazonSageMakerFullAccess` policy and the permissions to use Elastic Inference.
3.  Dummy ECR repo to run remote_function tests - :code:`aws ecr create-repository --repository-name remote-function-dummy-container`

**Running Specific Integration Tests:**
```bash
tox -- -k 'test_i_care_about'
```

**Running All Integration Tests:**
```bash
tox -- tests/integ
```

**Running Integration Tests in Parallel:**
```bash
tox -- -n auto tests/integ
```
## Contributing

For information on how to contribute, see the [Contributing Guidelines](CONTRIBUTING.md).