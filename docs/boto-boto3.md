# Boto3: The AWS SDK for Python

**Empower your Python applications to seamlessly interact with Amazon Web Services using Boto3, the official AWS SDK for Python.**  Learn more and contribute on [GitHub](https://github.com/boto/boto3).

[![Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

## Key Features of Boto3:

*   **Comprehensive AWS Service Support:** Access a wide range of AWS services like Amazon S3, Amazon EC2, DynamoDB, and many more.
*   **Simplified Development:** Write Python code that easily integrates with AWS services using Pythonic APIs.
*   **Credential Management:** Securely configure and manage your AWS credentials.
*   **Community-Driven:** Benefit from an active community and continuous updates from Amazon Web Services.
*   **Open Source:** Boto3 is an open-source project, allowing for community contributions and transparency.

## Getting Started

### Prerequisites

*   A supported version of Python.

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

    *Alternatively, you can install from source:*

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set up your AWS credentials:**  Configure your AWS access key ID and secret access key. The easiest way is to create a `~/.aws/credentials` file:

    ```ini
    [default]
    aws_access_key_id = YOUR_KEY
    aws_secret_access_key = YOUR_SECRET
    ```

2.  **Set a default region (optional):** Set your AWS region in the `~/.aws/config` file:

    ```ini
    [default]
    region=us-east-1
    ```

    *Find more credential configuration options [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).*

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

You can run tests using `tox`:

```bash
tox
```
Or run specific tests with `pytest`:

```bash
pytest tests/unit
```

## Help and Resources

*   **Documentation:** Access the latest [documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).
*   **Community Support:**
    *   Ask questions on [Stack Overflow](https://stackoverflow.com/) with the tag `boto3`.
    *   Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   **Report Issues:**  If you find a bug, [open an issue on GitHub](https://github.com/boto/boto3/issues/new).

## Contributing

We welcome contributions!  Please review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document before submitting pull requests or issues.

## Maintenance and Support

Boto3 follows the [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html).

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)