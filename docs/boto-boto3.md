# Boto3: The AWS SDK for Python

**Boto3 empowers Python developers to effortlessly interact with Amazon Web Services, simplifying cloud infrastructure management.**  Learn more and contribute at the [original Boto3 repository](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Seamless AWS Integration:** Interact with a wide range of AWS services, including Amazon S3, Amazon EC2, and many more.
*   **Simplified Development:** Write Python code to manage your AWS resources with ease.
*   **Comprehensive Documentation:** Access up-to-date documentation and examples for all supported services.
*   **Community-Driven:** Benefit from a vibrant community and contribute to the project.

## Getting Started

### Prerequisites

*   A supported version of Python is installed.

### Installation

1.  **Set up a virtual environment:**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

    or

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up your AWS credentials (e.g., in `~/.aws/credentials`):**

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set up a default region (e.g., in `~/.aws/config`):**

    ```ini
    [default]
    region=us-east-1
    ```

    Other credential configuration methods are described in the official [AWS documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Testing

Run tests using `tox`:

```bash
tox
```

You can also run tests with specific options, or for a specific Python version.

## Help and Support

*   **Stack Overflow:** Ask questions and find answers with the `boto3` tag: [https://stackoverflow.com/questions/tagged/boto3](https://stackoverflow.com/questions/tagged/boto3)
*   **AWS Support:** Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/)
*   **GitHub Issues:** Report bugs and feature requests: [https://github.com/boto/boto3/issues/new](https://github.com/boto/boto3/issues/new)

## Contributing

We welcome contributions! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting issues or pull requests.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase. For information about maintenance and support for SDK major versions and their underlying dependencies, see the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)
*   [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)