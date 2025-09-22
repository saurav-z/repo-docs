![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

# SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models

**Quickly build, train, and deploy your machine learning models on Amazon SageMaker with the open-source SageMaker Python SDK!**

[Explore the SageMaker Python SDK on GitHub](https://github.com/aws/sagemaker-python-sdk)

**Key Features:**

*   **Framework Support:** Train and deploy models using popular deep learning frameworks like Apache MXNet, TensorFlow, PyTorch, and others.
*   **Built-in Algorithms:** Utilize Amazon's scalable, optimized implementations of core machine learning algorithms.
*   **Custom Algorithms:** Bring your own algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Serving:** Deploy and perform predictions against SparkML models serialized with the MLeap library.
*   **Easy Deployment:** Deploy models with a few lines of code.
*   **Model Management:** Manage your models, endpoints, and training jobs with ease.
*   **Integration with other SageMaker Features**: Integrate with Automatic Model Tuning, Batch Transform, Model Monitoring, and more.
*   **Comprehensive Documentation:** Detailed documentation, including an API reference, available on [Read the Docs](https://sagemaker.readthedocs.io/en/stable/).

**Installation:**

```bash
pip install sagemaker  # Installs the latest version
```

or install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

**Supported Python Versions:**

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

**Additional Information:**

*   **AWS Permissions:**  Requires permissions as outlined in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
*   **Telemetry:**  The SDK has telemetry enabled (opt-out available). More information on [Read The Docs](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).
*   **Licensing:**  Licensed under the Apache 2.0 License.
*   **Testing:** Includes unit and integration tests.