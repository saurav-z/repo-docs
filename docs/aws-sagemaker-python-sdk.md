![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models with Ease

**Unlock the power of Amazon SageMaker with the Python SDK, enabling seamless training and deployment of your machine learning models.**  [Explore the original repository](https://github.com/aws/sagemaker-python-sdk).

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others.
*   **Amazon Algorithms:** Utilize scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy trained models for real-time or batch inference.
*   **Integration with MLeap:** Deploy SparkML models using the SageMaker SparkML Serving feature.

**Core Capabilities & Functionality**

*   **Training & Deployment:** Facilitate training and deployment of models.
*   **Model Hosting:** Easily host your models.
*   **Model Packages:** Consume SageMaker model packages.
*   **Automatic Model Tuning:** Utilize automatic model tuning capabilities.
*   **Batch Transform:** Leverage batch transform functionality.
*   **VPC Secure Training and Inference:** Secure training and inference with VPC.
*   **Inference Pipelines:** Simplify model inference workflows.
*   **Model Monitoring:** Monitor model performance and drift.
*   **Model Debugging:** Use built-in debugging tools.
*   **Processing Jobs:** Run data processing jobs.

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

### Supported Operating Systems & Python Versions

*   **Operating Systems:** Unix/Linux and Mac.
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12.

### AWS Permissions

Ensure your IAM role has the necessary permissions to use SageMaker. See the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.

### Testing

#### Unit Tests

Run unit tests using tox:

```bash
tox tests/unit
```

#### Integration Tests

Run integration tests with:

```bash
tox -- tests/integ
```

### Contributing

#### Git Hooks
Enable git hooks:

```bash
find .git/hooks -type l -exec rm {} \;
find .githooks -type f -exec ln -sf ../../{} .git/hooks/ \;
```

#### Building Sphinx Docs

1.  Set up a Python environment and install dependencies from `doc/requirements.txt`.
2.  Install the SDK locally using `pip install --upgrade .`.
3.  Navigate to the `doc` directory and run `make html`.
4.  Preview the site with a Python web server, visiting `http://localhost:8000`.

### Additional Information

*   **Telemetry:** Telemetry is enabled to help improve the SDK. Opt-out is available via the `TelemetryOptOut` configuration parameter.
*   **Licensing:** The SDK is licensed under the Apache 2.0 License.
*   **SageMaker SparkML Serving:** Deploy and perform predictions against SparkML models using the `MLeap` library.  See the `SageMaker SparkML Serving Container`_ for more information.