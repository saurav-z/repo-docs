# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Unlock the power of machine learning on Amazon SageMaker with the official Python SDK, designed for seamless model training, deployment, and management.**  ([View on GitHub](https://github.com/aws/sagemaker-python-sdk))

## Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms for SageMaker and GPU training.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms built into SageMaker-compatible Docker containers.
*   **Simplified Model Deployment:** Easily deploy trained models with a few lines of code.
*   **Model Management:**  Streamline your ML lifecycle with features like model versioning, monitoring, and automatic tuning.
*   **Comprehensive Documentation:** Access detailed documentation, including an API reference, via [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).
*   **SparkML Serving:** Deploy SparkML models serialized with MLeap for performing predictions in SageMaker.

## Getting Started

### Installing the SageMaker Python SDK

Install the latest version using pip:

```bash
pip install sagemaker
```

or, install from source:

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

## Additional Resources

*   **Using the SageMaker Python SDK**:  See the comprehensive overview:  [Overview](https://sagemaker.readthedocs.io/en/stable/overview.html)
*   **Documentation**:  Browse the full documentation:  [Read the Docs](https://sagemaker.readthedocs.io/)
*   **Examples and Tutorials:** Explore various training and deployment scenarios.
*   **SageMaker SparkML Serving:** Deploy SparkML models: [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)

## Telemetry

The `sagemaker` library has telemetry enabled to help us understand user needs and deliver new features.  You can opt-out by setting the `TelemetryOptOut` parameter to `true`.  See the documentation for details.

## AWS Permissions

Ensure your IAM role has the necessary permissions to interact with SageMaker.  Detailed information is available in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## License

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## Testing

Run unit tests with tox:

```bash
tox tests/unit
```

Run integration tests:

```bash
tox -- tests/integ
```

## Git Hooks

Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

## Building Sphinx Docs

Follow the instructions in the original README to build the documentation locally using Sphinx.

```
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt

pip install --upgrade .

cd sagemaker-python-sdk/doc
make html
```

```