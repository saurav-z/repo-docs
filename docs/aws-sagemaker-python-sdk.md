<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# SageMaker Python SDK: Simplify Machine Learning on AWS

**The SageMaker Python SDK provides a powerful and flexible way to build, train, and deploy machine learning models on Amazon SageMaker.**  [View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

---

## Key Features

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and more.
*   **Built-in Algorithms:** Leverage scalable implementations of Amazon's core machine learning algorithms, optimized for SageMaker.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Model Deployment:** Easily deploy trained models for real-time or batch predictions.
*   **Integration:** Seamlessly integrates with other SageMaker features like Automatic Model Tuning, Batch Transform, and Model Monitoring.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models serialized with MLeap.

---

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

### Supported Versions
*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

---
## Code Style
*   Uses Black code style for formatting.

---

## Documentation

For comprehensive documentation, including API references, visit the [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).

---
## Additional Information

*   **Telemetry:** The SDK collects telemetry data to improve user experience. You can opt out by configuring `TelemetryOptOut = true` in SDK defaults.
*   **AWS Permissions:**  Ensure your IAM role has the necessary permissions for SageMaker operations as outlined in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
*   **Licensing:** Licensed under the Apache 2.0 License (Copyright Amazon.com, Inc. or its affiliates).
*   **Testing:** Includes unit and integration tests.  Run with `tox tests/unit` for unit tests and `tox tests/integ` for integration tests.

---

## SageMaker SparkML Serving

Deploy and predict with your SparkML models using this functionality. More information can be found within the documentation.