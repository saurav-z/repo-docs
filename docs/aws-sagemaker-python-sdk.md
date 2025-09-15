# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**Easily build, train, and deploy your machine learning models with the open-source SageMaker Python SDK, streamlining your ML workflow on Amazon SageMaker.**  [View the original repository on GitHub](https://github.com/aws/sagemaker-python-sdk)

![SageMaker Banner](https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png)

The SageMaker Python SDK provides a flexible and powerful way to manage your entire machine learning lifecycle on Amazon SageMaker.  This library allows you to:

*   **Train Models with Popular Frameworks:** Supports Apache MXNet, TensorFlow, PyTorch, Chainer, Scikit-learn, and XGBoost.
*   **Utilize Amazon Algorithms:** Leverage scalable, optimized implementations of core machine learning algorithms.
*   **Bring Your Own Algorithms:** Train and deploy models using your custom algorithms within SageMaker-compatible Docker containers.
*   **Deploy SparkML Models:** Host and perform predictions against SparkML models serialized with the MLeap library.
*   **Simplified Deployment:** Easily deploy your trained models for real-time or batch inference.
*   **Integration with SageMaker Features:** Seamlessly integrate with SageMaker features such as Automatic Model Tuning, Batch Transform, Model Monitoring, and Debugger.

## Key Features

*   **Framework Support:** Train models using popular deep learning frameworks, including TensorFlow, PyTorch, and more.
*   **Algorithm Flexibility:** Use built-in SageMaker algorithms, your own custom algorithms, or models from other sources.
*   **Simplified Deployment:** Easily deploy models for real-time or batch inference.
*   **Model Management:** Streamline the management of your models with features like automatic model tuning and model monitoring.
*   **Integration with SageMaker:** Access to a wide range of SageMaker features like model monitoring, debugging, and processing.

## Installation

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

## Supported Platforms

*   **Operating Systems:** Unix/Linux and Mac
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12

## Documentation

For detailed documentation, API references, and tutorials, please visit the [Read the Docs](https://sagemaker.readthedocs.io/en/stable/) site.

## Additional Resources

*   [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) (Permissions)
*   [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) (Telemetry Opt-Out)
*   [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container) (SparkML Serving)
*   [MLeap](https://github.com/combust/mleap)

## Telemetry

The SDK has telemetry enabled to help improve the library.  You can opt-out by configuring the `TelemetryOptOut` parameter.  See the docs for details.

## Licensing

The SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).