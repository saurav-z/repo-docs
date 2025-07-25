<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  </a>
  <br/>
  <h1>SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease</h1>
  <p>The SageMaker Python SDK simplifies building, training, and deploying machine learning models on Amazon SageMaker.</p>
  <a href="https://github.com/aws/sagemaker-python-sdk">View the Project on GitHub</a>
</div>

---

## Key Features of the SageMaker Python SDK

*   **Framework Flexibility:** Train models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Seamlessly integrate custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Easy Deployment:** Deploy trained models with a few lines of code.
*   **Model Package Consumption:** Utilize and integrate with pre-built model packages.
*   **Advanced Features:** Supports SageMaker Automatic Model Tuning, Batch Transform, Inference Pipelines, Model Monitoring, and more.
*   **SparkML Serving:** Deploy and perform predictions against SparkML Models in SageMaker, using the MLeap library.
*   **Comprehensive Documentation:** Detailed documentation, including API references, is available at [Read the Docs](https://sagemaker.readthedocs.io).

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Python Versions

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Key Capabilities

*   **Training**: Utilize various algorithms and frameworks, including those from AWS and the ability to bring your own models.
*   **Deployment**: Deploy trained models with ease using the included tools.
*   **Model Monitoring**: Monitor deployed models for performance and drift.

## Running Tests

Install testing dependencies:
```bash
pip install --upgrade .[test]
```
Run unit tests:
```bash
tox tests/unit
```
Run integration tests:
```bash
tox -- tests/integ
```

## Important Information

### AWS Permissions

The SageMaker Python SDK requires the necessary AWS permissions to interact with SageMaker. For details on required permissions, refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).  If using an IAM role with a path, grant permission for `iam:GetRole`.

### Telemetry

The SDK includes telemetry to help improve the service. You can opt-out by setting the `TelemetryOptOut` parameter. See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for instructions.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. The license is available at: http://aws.amazon.com/apache2.0/

## Additional Resources

*   **Documentation:** [Read the Docs](https://sagemaker.readthedocs.io/)
*   **SageMaker SparkML Serving Container:** [https://github.com/aws/sagemaker-sparkml-serving-container](https://github.com/aws/sagemaker-sparkml-serving-container)