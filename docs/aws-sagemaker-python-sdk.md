<!-- Banner Image -->
<img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">

# Amazon SageMaker Python SDK: Simplify Machine Learning Development and Deployment

**The SageMaker Python SDK empowers data scientists and developers to build, train, and deploy machine learning models seamlessly within the Amazon SageMaker ecosystem.**  Explore the power of this open-source library on [GitHub](https://github.com/aws/sagemaker-python-sdk).

## Key Features of the SageMaker Python SDK

*   **Simplified Model Training:** Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, as well as Amazon algorithms.
*   **Flexible Deployment Options:** Deploy models using Amazon algorithms, custom algorithms in SageMaker compatible Docker containers, or your own models.
*   **Framework Support:** Built-in support for a wide range of frameworks:
    *   Apache MXNet
    *   TensorFlow
    *   PyTorch
    *   Chainer
    *   Scikit-learn
    *   XGBoost
*   **Scalable Machine Learning:** Leverage Amazon SageMaker's infrastructure for scalable training and deployment.
*   **Model Tuning and Management:** Utilize features like automatic model tuning, batch transform, and model monitoring.
*   **Integration:** Integrate with other SageMaker services, including processing, debugger, and autopilot.
*   **SparkML Serving:** Seamlessly perform predictions against SparkML models serialized with MLeap.

## Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker==<Latest version from pyPI from https://pypi.org/project/sagemaker/>
```

Alternatively, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

## Core Functionality

*   **Training and Deployment:** This SDK allows you to orchestrate the training and deployment of your machine learning models on Amazon SageMaker.
*   **Framework Integration:** Support for popular machine learning frameworks makes it easy to get started with SageMaker.
*   **Algorithm Support:** Use Amazon's built-in algorithms or bring your own custom algorithms.
*   **Model Serving:** Deploy trained models as endpoints for real-time inference, or use batch transform for large-scale predictions.

## Supported Python Versions

The SageMaker Python SDK is tested on:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

## Running Tests

To run unit tests:

```bash
tox tests/unit
```

To run integration tests:

```bash
tox -- tests/integ
```

(You may need to set up AWS credentials and an IAM role as described in the original README to run integration tests.)

## Additional Resources

*   **Documentation:** [Read the Docs](https://sagemaker.readthedocs.io/)
*   **SageMaker SparkML Serving Container:** [https://github.com/aws/sagemaker-sparkml-serving-container](https://github.com/aws/sagemaker-sparkml-serving-container)

## License

This SDK is licensed under the Apache 2.0 License.

---

**[Back to Top](#)**