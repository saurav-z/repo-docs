# Boto3: The AWS SDK for Python - Simplify AWS Service Interactions

Boto3 is the official AWS SDK for Python, empowering developers to build robust applications that seamlessly interact with Amazon Web Services.  Explore the power of Boto3 and [discover the full documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html).

[![Package Version](http://img.shields.io/pypi/v/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![Python Versions](https://img.shields.io/pypi/pyversions/boto3.svg?style=flat)](https://pypi.python.org/pypi/boto3/)
[![License](http://img.shields.io/pypi/l/boto3.svg?style=flat)](https://github.com/boto/boto3/blob/develop/LICENSE)

**Key Features:**

*   **Comprehensive AWS Service Support:**  Interact with a wide range of AWS services, including Amazon S3, Amazon EC2, and many more.
*   **Simplified API:** Provides an intuitive and Pythonic interface for managing AWS resources.
*   **Authentication and Authorization:**  Handles AWS credentials securely, simplifying access to your resources.
*   **Well-Documented:**  Benefit from extensive documentation, examples, and tutorials to get you started quickly.
*   **Active Community:**  Leverage a thriving community for support, troubleshooting, and contributions.

## Getting Started

### Prerequisites

*   Python (Ensure you have a supported version, see notices below)

### Installation

1.  **Create a Virtual Environment (Recommended):**

    ```bash
    python -m venv .venv
    . .venv/bin/activate
    ```

2.  **Install Boto3 from PyPI:**

    ```bash
    python -m pip install boto3
    ```

    Or, install from source:

    ```bash
    git clone https://github.com/boto/boto3.git
    cd boto3
    python -m pip install -r requirements.txt
    python -m pip install -e .
    ```

### Configuration

1.  **Set Up AWS Credentials:**  Configure your AWS credentials using either:

    *   **Credentials File:** (e.g., `~/.aws/credentials`)

        ```ini
        [default]
        aws_access_key_id = YOUR_KEY
        aws_secret_access_key = YOUR_SECRET
        ```

    *   **Configuration File:** (e.g., `~/.aws/config`)

        ```ini
        [default]
        region=us-east-1
        ```

    *  Find other credential configuration methods [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)

### Example Usage

```python
import boto3
s3 = boto3.resource('s3')
for bucket in s3.buckets.all():
    print(bucket.name)
```

## Running Tests

You can run tests for various Python versions using `tox`:

```bash
tox
```

Run tests for specific environments or test files:

```bash
tox -- unit/test_session.py
tox -e py26,py33 -- integration/
pytest tests/unit
```

## Notices

**Python 3.8 Support:**

*   Support for Python 3.8 ended on 2025-04-22. Please refer to the [Python Software Foundation end of support <https://peps.python.org/pep-0569/#lifespan>`__ and [this blog post <https://aws.amazon.com/blogs/developer/python-support-policy-updates-for-aws-sdks-and-tools/>]`__ for more information.

## Getting Help

*   **Stack Overflow:** Ask questions tagged with [boto3](https://stackoverflow.com/questions/tagged/boto3).
*   **AWS Support:**  Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   **GitHub Issues:** Report bugs and feature requests on the [GitHub repository](https://github.com/boto/boto3/issues/new).

## Contributing

We welcome contributions!  Review the [CONTRIBUTING](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst) document for guidelines.

## Maintenance and Support

Boto3 was made generally available on 06/22/2015 and is currently in the full support phase of the availability life cycle. For maintenance and support details, see the following in the AWS SDKs and Tools Shared Configuration and Credentials Reference Guide:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## More Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)
*   **[Boto3 GitHub Repository](https://github.com/boto/boto3)**