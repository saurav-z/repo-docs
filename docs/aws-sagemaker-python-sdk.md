[![SageMaker Python SDK Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)](https://github.com/aws/sagemaker-python-sdk)

# SageMaker Python SDK: Simplify Machine Learning on AWS

**The SageMaker Python SDK is your key to building, training, and deploying machine learning models on Amazon SageMaker with ease.**  This open-source library streamlines the end-to-end ML lifecycle, allowing you to focus on innovation. Explore the [original repo](https://github.com/aws/sagemaker-python-sdk) for more details.

## Key Features

*   **Framework Flexibility:** Supports popular deep learning frameworks like Apache MXNet and TensorFlow, along with PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Built-in Algorithms:** Access scalable, optimized Amazon algorithms for efficient training.
*   **Bring Your Own Algorithms:** Seamlessly integrate your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Simplified Model Deployment:** Deploy trained models with just a few lines of code.
*   **Comprehensive Documentation:**  Access detailed documentation, including an API reference, at [Read the Docs](https://sagemaker.readthedocs.io).
*   **Model Monitoring:** Includes Model Monitoring for insights into your model's performance.

## Core Functionality

The SageMaker Python SDK empowers you to:

*   **Train Models:** Leverage a wide range of frameworks and algorithms for model training.
*   **Deploy Models:** Easily deploy your trained models for real-time or batch inference.
*   **Manage the ML Lifecycle:** Streamline the entire machine learning workflow, from data preparation to model monitoring.

## Installation

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Supported Versions

*   **Python:** 3.9, 3.10, 3.11, 3.12
*   **Operating Systems:** Unix/Linux and Mac

## SageMaker SparkML Serving

Integrate SparkML models into SageMaker with the SparkML Serving functionality. Deploy models serialized with the MLeap library for efficient predictions.

## Additional Information

*   **AWS Permissions:** Learn about the necessary AWS permissions in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).
*   **Licensing:**  Apache 2.0 License.
*   **Testing:** Run unit and integration tests using tox.

    ```bash
    tox tests/unit
    ```

    ```bash
    tox -- tests/integ
    ```

*   **Git Hooks:** Automate tasks with Git hooks.
*   **Building Sphinx Docs:**  Build documentation using Sphinx.