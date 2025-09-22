# Boto3: The AWS SDK for Python - Simplify Cloud Interactions

Boto3 empowers Python developers to effortlessly interact with Amazon Web Services, streamlining cloud application development.  [Get started with Boto3 on GitHub!](https://github.com/boto/boto3)

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Simplified AWS Service Access:** Interact with services like Amazon S3, Amazon EC2, and many more using Python.
*   **Comprehensive Documentation:** Access detailed documentation on the [official Boto3 documentation site](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
*   **Easy Installation:** Quickly install Boto3 using pip.
*   **Flexible Configuration:** Configure credentials and regions effortlessly for secure access to AWS resources.
*   **Active Community & Support:** Benefit from a thriving community and readily available resources for assistance.

## Getting Started

### Prerequisites

*   Python (See supported versions on the shields above)

### Installation

1.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3:**

    ```bash
    python -m pip install boto3
    ```

### Configuration

1.  **Configure AWS Credentials:**  Set up your AWS credentials in your environment or in the `~/.aws/credentials` file:

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set a Default Region (Optional):** Configure your default AWS region in `~/.aws/config`:

    ```ini
    [default]
    region=us-east-1
    ```

    For more details, see the [AWS Credentials documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Basic Usage Example

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

Tests can be executed using `tox`:

```bash
tox
```

## Getting Help

*   **Stack Overflow:** Ask questions tagged with `boto3` on [Stack Overflow](https://stackoverflow.com/questions/tagged/boto3).
*   **AWS Support:** Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   **GitHub Issues:** Report bugs or feature requests on [GitHub Issues](https://github.com/boto/boto3/issues/new).

## Contributing

Contributions are welcome! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting pull requests or issues.

## Maintenance and Support

Boto3 is currently in the full support phase of its availability lifecycle.  Refer to the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide for details:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)