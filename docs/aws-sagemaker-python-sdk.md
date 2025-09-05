<!-- Banner Image for SEO and Visual Appeal -->
<p align="center">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker Banner" width="400">
</p>

<!-- Badges for quick access -->
<p align="center">
  <a href="https://pypi.org/project/sagemaker" target="_blank"><img src="https://img.shields.io/pypi/v/sagemaker.svg" alt="PyPI Latest Version"></a>
  <a href="https://anaconda.org/conda-forge/sagemaker-python-sdk" target="_blank"><img src="https://img.shields.io/conda/vn/conda-forge/sagemaker-python-sdk.svg" alt="Conda-Forge Version"></a>
  <a href="https://pypi.org/project/sagemaker" target="_blank"><img src="https://img.shields.io/pypi/pyversions/sagemaker.svg" alt="Supported Python Versions"></a>
  <a href="https://github.com/python/black" target="_blank"><img src="https://img.shields.io/badge/code_style-black-000000.svg" alt="Code style: black"></a>
  <a href="https://sagemaker.readthedocs.io/en/stable/" target="_blank"><img src="https://readthedocs.org/projects/sagemaker/badge/?version=stable" alt="Documentation Status"></a>
  <a href="https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml" target="_blank"><img src="https://github.com/aws/sagemaker-python-sdk/actions/workflows/codebuild-ci-health.yml/badge.svg" alt="CI Health"></a>
</p>

# SageMaker Python SDK: Train, Deploy, and Manage Machine Learning Models with Ease

The SageMaker Python SDK simplifies the entire machine learning lifecycle, enabling data scientists and developers to build, train, and deploy models efficiently on Amazon SageMaker. Explore the power of the [SageMaker Python SDK](https://github.com/aws/sagemaker-python-sdk) and unlock the potential of your data.

## Key Features

*   **Model Training**: Train models using popular deep learning frameworks like Apache MXNet and TensorFlow, along with Amazon's optimized built-in algorithms. Supports custom algorithms within SageMaker compatible Docker containers.
*   **Model Deployment**: Deploy trained models with ease, making them accessible for real-time or batch predictions.
*   **Framework Support**: Comprehensive support for popular frameworks including TensorFlow, PyTorch, MXNet, Chainer, Scikit-learn, and XGBoost.
*   **Scalability**: Leverage Amazon SageMaker's infrastructure for scalable training and deployment, including GPU support.
*   **Integration**: Seamless integration with other AWS services for a comprehensive machine learning platform.
*   **SparkML Serving**: Perform predictions against a SparkML Model in SageMaker using the MLeap library.

## Core Functionality

*   **Training with Built-in Algorithms**: Utilize Amazon SageMaker's pre-built, optimized algorithms for common machine learning tasks.
*   **Bring Your Own Algorithms**: Easily integrate custom algorithms packaged in Docker containers.
*   **Hyperparameter Tuning**: Automate the process of finding optimal model hyperparameters using SageMaker's tuning capabilities.
*   **Model Monitoring**: Monitor model performance in production to identify and address issues quickly.
*   **Model Debugging**: Leverage SageMaker Debugger to identify and resolve model training issues efficiently.
*   **Batch Transform**: Process large datasets efficiently with batch transformation jobs.
*   **Inference Pipelines**: Chain together multiple models for complex inference workflows.

## Getting Started

### Installation

Install the latest version of the SageMaker Python SDK using pip:

```bash
pip install sagemaker
```

Alternatively, install from source:

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

## Documentation

For detailed information, examples, and API references, visit the [official documentation](https://sagemaker.readthedocs.io/en/stable/).

## Contributing

Contributions are welcome! Please review our [CONTRIBUTING.md](https://github.com/aws/sagemaker-python-sdk/blob/master/CONTRIBUTING.md) for guidelines.

## License

SageMaker Python SDK is licensed under the [Apache 2.0 License](http://aws.amazon.com/apache2.0/).