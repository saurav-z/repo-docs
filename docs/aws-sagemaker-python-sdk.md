# Amazon SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models

**Supercharge your machine learning workflow on Amazon SageMaker with the official Python SDK!** ([Original Repo](https://github.com/aws/sagemaker-python-sdk))

This open-source library empowers data scientists and developers to seamlessly train, deploy, and manage machine learning models on Amazon SageMaker.  Simplify your ML lifecycle and accelerate your projects with a comprehensive set of tools.

## Key Features:

*   **Framework Agnostic:** Supports popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage optimized, scalable implementations of core machine learning algorithms.
*   **Custom Algorithm Support:** Train and deploy models using your own algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Training:**  Easily configure and launch training jobs with minimal code.
*   **Flexible Deployment:** Deploy models to production with various instance types and endpoint configurations.
*   **Model Management:** Streamline model versioning, monitoring, and updates.
*   **Integration with SparkML:** Deploy and perform predictions against SparkML models using the MLeap library.
*   **Automatic Model Tuning**:  Automate hyperparameter optimization to find the best performing model.
*   **Batch Transform**:  Perform batch predictions on large datasets efficiently.
*   **Model Monitoring**: Track model performance and detect issues in real-time.
*   **Extensive Documentation:** Comprehensive documentation and examples for quick onboarding.

## Getting Started:

### Installation

Install the latest version from PyPI:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Python Versions:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Learn More:

*   **Documentation:**  [Read the Docs](https://sagemaker.readthedocs.io/)
*   **SageMaker SparkML Serving Container:** [https://github.com/aws/sagemaker-sparkml-serving-container](https://github.com/aws/sagemaker-sparkml-serving-container)

## Additional Resources:

*   **Licensing:** Apache 2.0 License ([http://aws.amazon.com/apache2.0/](http://aws.amazon.com/apache2.0/))
*   **AWS Permissions:** [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
*   **Telemetry:** The `sagemaker` library has telemetry enabled.  Opt-out by setting the `TelemetryOptOut` parameter to `true` in the SDK defaults configuration.  See [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for detailed instructions.
*   **Testing:**  Detailed instructions are available to test the SDK.
*   **Building Docs:** Detailed instructions for building the Sphinx docs.
*   **Git Hooks:** Instructions to enable git hooks.