# Boto3: The AWS SDK for Python

**Easily interact with Amazon Web Services (AWS) using Python with the Boto3 SDK.**  Find the original repo [here](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

Boto3 is the official AWS SDK for Python, allowing developers to seamlessly integrate AWS services like Amazon S3 and Amazon EC2 into their Python applications.  It simplifies the process of interacting with AWS, enabling you to manage resources, automate tasks, and build powerful cloud-based solutions.

## Key Features

*   **Comprehensive AWS Service Support:** Access a wide range of AWS services.
*   **Simplified API:**  Provides a Pythonic interface for interacting with AWS services.
*   **Credential Management:** Securely configure and manage your AWS credentials.
*   **Easy to Install:** Simple installation via `pip`.
*   **Well-Documented:** Extensive documentation to help you get started quickly.

## Getting Started

### Prerequisites

*   Python (see the Python versions supported by the shields above)

### Installation

1.  **Set up your environment (optional but recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3:**

    ```bash
    python -m pip install boto3
    ```
    
    or install from source:
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

2.  **Configure your default region (e.g., in `~/.aws/config`):**

    ```ini
    [default]
    region=us-east-1
    ```

    Other credential configuration methods can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

You can run tests in all supported Python versions using `tox`:

```bash
tox
```

## Getting Help

*   Ask a question on [Stack Overflow](https://stackoverflow.com/) and tag it with `boto3`.
*   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   If you believe you've found a bug, please [open an issue](https://github.com/boto/boto3/issues/new).

## Contributing

We welcome contributions! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting issues or pull requests.

## Maintenance and Support

Boto3 is currently in the full support phase. See the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)
*   [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)