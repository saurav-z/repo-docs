<!-- Banner Image -->
<a href="https://github.com/aws/sagemaker-python-sdk">
  <img src="https://github.com/aws/sagemaker-python-sdk/raw/master/branding/icon/sagemaker-banner.png" alt="SageMaker" height="100">
</a>

# SageMaker Python SDK: Train and Deploy Machine Learning Models on AWS

**The SageMaker Python SDK empowers data scientists and developers to build, train, and deploy machine learning models seamlessly within Amazon SageMaker.**

## Key Features

*   **Framework Support:** Train models with popular deep learning frameworks like Apache MXNet and TensorFlow.
*   **Amazon Algorithms:** Utilize scalable, optimized Amazon algorithms for core machine learning tasks.
*   **Bring Your Own Algorithms:** Train and host models using your custom algorithms packaged in SageMaker-compatible Docker containers.
*   **Easy Deployment:** Simplify model deployment with pre-built containers and flexible instance types.
*   **Model Management:** Manage model versions, track experiments, and monitor model performance.
*   **Integration with MLeap:** Deploy SparkML models serialized with the MLeap library for prediction.
*   **Extensive Documentation:** Access detailed documentation, including API references, for comprehensive guidance.
*   **Telemetry Opt-Out:**  Control your telemetry preferences.

## Getting Started

### Installation

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

### Supported Versions

*   **Operating Systems:** Unix/Linux and Mac
*   **Python:** 3.9, 3.10, 3.11, 3.12

## Core Functionality

*   **Training:** Train models using various frameworks and algorithms on SageMaker.
*   **Deployment:** Deploy trained models for real-time or batch inference.
*   **Model Management:** Manage model versions, track training runs, and monitor model performance.
*   **Processing:** Perform pre- and post-processing tasks on your data.
*   **Model Debugging:** Utilize the SageMaker Debugger to identify and resolve issues in your models.
*   **Automatic Model Tuning:** Automate the process of hyperparameter optimization.

## Usage Examples

Explore the SageMaker Python SDK through the `Read the Docs <https://sagemaker.readthedocs.io/en/stable/>` for detailed API references, guides, and example code.

## Telemetry

Telemetry is enabled by default to help improve the SDK. To opt-out, configure the  `TelemetryOptOut` parameter to `true`.  See `Configuring and using defaults with the SageMaker Python SDK <https://sagemaker.readthedocs.io/en/stable/overview.html#configuring-and-using-defaults-with-the-sagemaker-python-sdk>` for instructions.

## AWS Permissions

For information on required permissions, see the `AWS Documentation <https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html>`

## Licensing

This project is licensed under the Apache 2.0 License.

## Contributing

Contributions are welcome!

## Additional Resources

*   **Documentation:** `Read the Docs <https://sagemaker.readthedocs.io/en/stable/>`
*   **Source Code:** [SageMaker Python SDK GitHub Repository](https://github.com/aws/sagemaker-python-sdk)
*   **SparkML Serving Container:** `https://github.com/aws/sagemaker-sparkml-serving-container`

```

Key improvements and explanations:

*   **SEO Optimization:**  Included relevant keywords like "SageMaker," "machine learning," "train," "deploy," and "AWS" in the title and throughout the description.
*   **Concise Hook:** The one-sentence hook immediately grabs the reader's attention and summarizes the SDK's core purpose.
*   **Clear Headings:** Organized the information with clear, descriptive headings for better readability and SEO.
*   **Bulleted Key Features:**  Highlights the main benefits and features in an easily digestible format.
*   **Actionable Installation:** Includes the exact `pip install sagemaker` command for easy setup.
*   **Simplified & Condensed Content:**  Removed redundant information and streamlined the descriptions.
*   **Removed Redundant Links:**  Eliminated excessive links to the same documentation and GitHub repo.
*   **Added Contributing Note:**  This encourages community participation.
*   **Telemetry section improved**: Summarized and provides a link to the documentation.
*   **Reorganized sections**: Grouped information logically.
*   **Removed the table of contents** Kept the main categories.
*   **Git Hooks Section Removed:** Git Hooks are not essential to the product's purpose.
*   **Sphinx docs removed:**  This is primarily a developer's function.
*   **SparkML serving improved**:  Simplified to include only the essentials.
*   **Code blocks formatted:**  Added formatting for readability.
*   **Added a banner Image:** At the top, for a great visual experience