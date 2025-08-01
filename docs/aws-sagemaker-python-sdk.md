<div align="center">
  <a href="https://github.com/aws/sagemaker-python-sdk">
    <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
  </a>
  <h1>SageMaker Python SDK: Train and Deploy Machine Learning Models</h1>
</div>

The SageMaker Python SDK provides a flexible and powerful way to train and deploy your machine learning models on Amazon SageMaker.

**Key Features:**

*   **Train with Popular Frameworks:** Supports Apache MXNet, TensorFlow, PyTorch, and more.
*   **Utilize Amazon Algorithms:** Leverage optimized, scalable implementations of core ML algorithms.
*   **Bring Your Own Algorithms (BYOA):** Train and deploy models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **SparkML Serving:** Easily deploy and perform predictions with your SparkML models using the MLeap library.
*   **Extensive Functionality:** Includes support for automatic model tuning, batch transform, model monitoring, and more.

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

### Supported Platforms and Python Versions

*   **Operating Systems:** Unix/Linux, Mac
*   **Python Versions:** 3.9, 3.10, 3.11, and 3.12

## Key Features and Functionality

*   [Using SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html): A general overview and introduction to the SDK.
*   **Framework-Specific Tutorials:** Detailed guides for using popular frameworks like MXNet, TensorFlow, PyTorch, Scikit-learn, and XGBoost.
    *   [Using MXNet](https://sagemaker.readthedocs.io/en/stable/using_mxnet.html)
    *   [Using TensorFlow](https://sagemaker.readthedocs.io/en/stable/using_tf.html)
    *   [Using PyTorch](https://sagemaker.readthedocs.io/en/stable/using_pytorch.html)
    *   [Using Scikit-learn](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)
    *   [Using XGBoost](https://sagemaker.readthedocs.io/en/stable/using_xgboost.html)
*   [Amazon SageMaker Built-in Algorithm Estimators](src/sagemaker/amazon/README.rst):  Documentation for using the built-in algorithms.
*   [SageMaker SparkML Serving](https://github.com/aws/sagemaker-sparkml-serving-container): Deploy and perform predictions with your SparkML models using the MLeap library.
*   **Advanced Features:**
    *   [SageMaker Automatic Model Tuning](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-automatic-model-tuning)
    *   [SageMaker Batch Transform](https://sagemaker.readthedocs.io/en/stable/overview.html#sagemaker-batch-transform)
    *   [Model Monitoring](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_model_monitoring.html)
    *   [SageMaker Debugger](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_debugger.html)
    *   [SageMaker Processing](https://sagemaker.readthedocs.io/en/stable/amazon_sagemaker_processing.html)
    *   [Inference Pipelines](https://sagemaker.readthedocs.io/en/stable/overview.html#inference-pipelines)
    *   [BYO Model](https://sagemaker.readthedocs.io/en/stable/overview.html#byo-model)
    *   [SageMaker Autopilot](src/sagemaker/automl/README.rst)
    *   [Secure Training and Inference with VPC](https://sagemaker.readthedocs.io/en/stable/overview.html#secure-training-and-inference-with-vpc)

## Telemetry

The SDK collects telemetry data to improve the user experience.  You can opt-out by setting the `TelemetryOptOut` parameter in the SDK defaults configuration.  See the documentation for [Configuring and using defaults with the SageMaker Python SDK](https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk) for more details.

## AWS Permissions

SageMaker performs operations on your behalf.  For more information on the required permissions, please see the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html).

## Licensing

The SageMaker Python SDK is licensed under the Apache 2.0 License.

## Contributing and Testing

Contributions are welcome!  See the [original repository](https://github.com/aws/sagemaker-python-sdk) for testing and contribution guidelines.

### Running Tests

Install testing dependencies:
```bash
pip install --upgrade .[test]
```

or, for Zsh users:

```bash
pip install --upgrade .\[test]
```

*   **Unit Tests:** Run with tox:  `tox tests/unit`
*   **Integration Tests:** Run with tox: `tox tests/integ`

## Building Documentation

1.  **Setup:**
    ```bash
    conda create -n sagemaker python=3.12
    conda activate sagemaker
    conda install sphinx=5.1.1 sphinx_rtd_theme=0.5.0
    pip install -r doc/requirements.txt
    pip install --upgrade .
    ```
2.  **Build:** `cd sagemaker-python-sdk/doc && make html`
3.  **View:**  `cd _build/html && python -m http.server 8000` (then visit http://localhost:8000)