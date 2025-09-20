# Boto3: The AWS SDK for Python

**Boto3 empowers Python developers to seamlessly interact with Amazon Web Services, simplifying cloud infrastructure management.**  [Explore the original repository](https://github.com/boto/boto3).

[![Package Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features

*   **Simplified AWS Interaction:** Provides an easy-to-use interface for working with various AWS services, such as Amazon S3, Amazon EC2, and more.
*   **Pythonic API:** Designed with Python developers in mind, offering a familiar and intuitive API.
*   **Service Support:** Supports a wide range of AWS services, with new services and features constantly being added.  See the [documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for a complete list.
*   **Credential Management:**  Supports multiple methods for configuring your AWS credentials, including environment variables, configuration files, and IAM roles.

## Getting Started

### Prerequisites

*   A supported version of Python installed.

### Installation

1.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    python -m pip install boto3
    ```

    *Alternatively, install from source:*

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up AWS credentials:**

    ```ini
    # ~/.aws/credentials
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Configure a default region:**

    ```ini
    # ~/.aws/config
    [default]
    region=us-east-1
    ```

    *Other credential configuration methods can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)*

### Usage Example

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

```bash
# Run all tests
tox

# Run specific tests
tox -- unit/test_session.py

# Run tests with specific Python versions
tox -e py26,py33 -- integration/

# Run individual tests with your default Python version
pytest tests/unit
```

## Getting Help

*   **GitHub Issues:**  Report bugs and request features on [GitHub](https://github.com/boto/boto3/issues/new).
*   **Stack Overflow:** Ask questions and find answers on [Stack Overflow](https://stackoverflow.com/) and tag with `boto3`.
*   **AWS Support:** Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).

## Contributing

We welcome contributions! Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting any issues or pull requests.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase of the availability life cycle.  See the [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html) for more information.

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)