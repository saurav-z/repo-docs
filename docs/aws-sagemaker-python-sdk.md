<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="Logo" height="80">
  </a>

  <h3 align="center">Amazon SageMaker Python SDK</h3>

  <p align="center">
    <b>Simplify your Machine Learning workflow with the SageMaker Python SDK!</b> 
    <br />
    <a href="https://github.com/aws/sagemaker-python-sdk"><strong>Explore the Docs »</strong></a>
    <br />
    <br />
  </p>
</div>

## About the Project

The Amazon SageMaker Python SDK is an open-source library designed to streamline the training and deployment of machine learning models on Amazon SageMaker. This SDK provides a user-friendly interface for managing all aspects of your machine learning lifecycle, from data preparation and model training to deployment and monitoring.

Key Features:

*   **Framework Support:** Train and deploy models using popular deep learning frameworks such as Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Leverage scalable implementations of core machine learning algorithms optimized for SageMaker and GPU training.
*   **Custom Algorithms:** Easily integrate your own algorithms built into SageMaker-compatible Docker containers.
*   **Model Deployment:** Simplify the deployment of your models with built-in deployment capabilities.
*   **MLOps Capabilities**: Leverage SageMaker's MLOps capabilities such as Model Monitoring, Debugger, Processing and more.

### Getting Started

To get started, install the SDK using pip:

```bash
pip install sagemaker
```

For detailed installation instructions, including source installation, see the detailed guide in the [original repository](https://github.com/aws/sagemaker-python-sdk).

### Core Functionality:

*   **Training:** Train models using various frameworks, Amazon algorithms, and custom containers.
*   **Deployment:** Deploy trained models for real-time or batch inference.
*   **Model Management:** Manage model versions, track experiments, and monitor performance.
*   **Data Processing:** Utilize SageMaker Processing for data preparation and feature engineering.
*   **Model Tuning:** Perform automatic model tuning to optimize hyperparameters.

## Using the SageMaker Python SDK

The SDK provides tools for various aspects of the ML lifecycle:

*   **Training and Deployment**: See the [Read the Docs](https://sagemaker.readthedocs.io/en/stable/overview.html) for an overview of training and deployment.
*   **Framework-Specific Guides:** Explore guides for MXNet, TensorFlow, PyTorch, Scikit-learn, XGBoost, Chainer and Reinforcement Learning Estimators:
    *   [MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
    *   [TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
    *   [Chainer](https://sagemaker.readthedocs.io/en/stable/using_chainer.html)
    *   [PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
    *   [Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)
    *   [XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)
    *   [Reinforcement Learning](https://sagemaker.readthedocs.io/en/stable/using_rl.html)
*   **Advanced Features:**
    *   [SageMaker SparkML Serving](https://github.com/aws/sagemaker-sparkml-serving-container)
    *   [Amazon SageMaker Built-in Algorithm Estimators](src/sagemaker/amazon/README.rst)
    *   [SageMaker Autopilot](src/sagemaker/automl/README.rst)
    *   [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
    *   [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
    *   [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)
    *   [SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)
    *   [Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)

## Important Information

### Supported Operating Systems

The SageMaker Python SDK supports Unix/Linux and Mac operating systems.

### Supported Python Versions

The SageMaker Python SDK is tested on the following Python versions:

*   Python 3.9
*   Python 3.10
*   Python 3.11
*   Python 3.12

### Telemetry

The SDK has telemetry enabled to understand user needs. You can opt-out by setting the ``TelemetryOptOut`` parameter.  Learn more in the [official documentation](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk).

### AWS Permissions

Ensure your AWS Identity and Access Management (IAM) role has the necessary permissions for SageMaker. Read more in the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).  If using an IAM role with a path, grant permission for ``iam:GetRole``.

### Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License. See the full license at http://aws.amazon.com/apache2.0/.

### Running Tests

Run unit tests and integration tests to ensure your code works as expected.

1.  **Install Test Dependencies:**  Run `pip install --upgrade .[test]`
2.  **Unit Tests:** Use tox: `tox tests/unit`
3.  **Integration Tests:** Configure AWS credentials and IAM role, then run: `tox -- tests/integ` (or use options for parallel or selective execution)

### Contributing

Contributions are welcome! See the [Contribution Guidelines](CONTRIBUTING.md) for more information.

### Build the documentation

Setup a Python environment, install the dependencies listed in ``doc/requirements.txt``:

```bash
# conda
conda create -n sagemaker python=3.12
conda activate sagemaker
conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0

# pip
pip install -r doc/requirements.txt
```

Clone/fork the repo, and install your local version:

```bash
pip install --upgrade .
```

Then ``cd`` into the ``sagemaker-python-sdk/doc`` directory and run:

```bash
make html
```

You can edit the templates for any of the pages in the docs by editing the .rst files in the ``doc`` directory and then running ``make html`` again.

Preview the site with a Python web server:

```bash
cd _build/html
python -m http.server 8000
```

View the website by visiting http://localhost:8000