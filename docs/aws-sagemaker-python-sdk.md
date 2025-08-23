<!-- Banner Image (Retained for visual appeal, but not critical for SEO) -->
<!-- <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100"> -->

# Amazon SageMaker Python SDK: Train and Deploy Machine Learning Models

**The Amazon SageMaker Python SDK simplifies the entire machine learning workflow, from model training to deployment.**

[View the Original Repository](https://github.com/aws/sagemaker-python-sdk)

## Key Features:

*   **Simplified Model Training:** Train models with popular deep learning frameworks like Apache MXNet and TensorFlow, as well as Amazon algorithms, and custom algorithms in Docker containers.
*   **Flexible Deployment Options:** Easily deploy your trained models to Amazon SageMaker for real-time or batch inference.
*   **Framework Agnostic:** Supports various frameworks like Apache MXNet, TensorFlow, Chainer, PyTorch, Scikit-learn, and XGBoost.
*   **Integration with SageMaker Features:** Seamlessly integrates with SageMaker's advanced features such as Automatic Model Tuning, Batch Transform, and Model Monitoring.
*   **Scalable and Optimized:** Leverages SageMaker's infrastructure for scalable training and optimized performance.
*   **Telemetry Enabled:**  Helps you understand user needs, diagnose issues, and deliver new features.  Opt-out available.
*   **Comprehensive Documentation:** Detailed documentation, including API reference, available at [Read the Docs](https://sagemaker.readthedocs.io).
*   **SparkML Serving:** Support to perform predictions against a SparkML Model in SageMaker.

## Getting Started

### Installation

Install the latest version using pip:

```bash
pip install sagemaker
```

Or, install from source:

```bash
git clone https://github.com/aws/sagemaker-python-sdk.git
cd sagemaker-python-sdk
pip install .
```

### Supported Operating Systems and Python Versions:

*   **Operating Systems:** Unix/Linux, Mac
*   **Python Versions:** 3.9, 3.10, 3.11, 3.12

## Additional Information

*   **AWS Permissions:**  Requires the necessary permissions to use SageMaker.  See [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html) for details.  May require `iam:GetRole` if using an IAM role with a path.
*   **Licensing:**  Licensed under the Apache 2.0 License. Copyright Amazon.com, Inc. or its affiliates.
*   **Testing:**  Includes unit and integration tests. Instructions for running tests are provided in the original README.
*   **Documentation:**  Build documentation using Sphinx. Instructions for building and viewing the documentation are provided in the original README.
*   **SageMaker SparkML Serving:**  Enables predictions against SparkML models using the MLeap library.  Supports Spark 3.3. For more information see  [SageMaker SparkML Serving Container](https://github.com/aws/sagemaker-sparkml-serving-container)
```

**Key Improvements and SEO Optimizations:**

*   **Concise Hook:** A clear and concise opening sentence that immediately describes the SDK's purpose.
*   **Keyword Rich:** Uses relevant keywords like "Amazon SageMaker," "machine learning," "model training," and "deployment" throughout the description.
*   **Structured Headings:** Uses clear and descriptive headings for better readability and SEO.
*   **Bulleted Key Features:** Highlights the core functionalities in an easy-to-scan format.
*   **Clear Installation Instructions:** Provides simple and direct installation instructions.
*   **Updated Information:** Clarifies the latest supported python versions.
*   **Includes Links:** Links back to the original repository and other important resources like the documentation and AWS documentation.
*   **Content Organization:**  Organized the information logically for both users and search engines.
*   **Removes Redundancy:**  Removed redundant information present in the original README, focusing on the most important and relevant details.
*   **Emphasis on Benefits:** Highlights the *benefits* of using the SDK (simplified workflow, flexible deployment, scalability).
*   **Calls to Action (Implied):** Encourages users to install and explore the SDK.