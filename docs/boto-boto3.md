# Boto3: The AWS SDK for Python

**Effortlessly interact with Amazon Web Services using Python with the powerful and versatile Boto3 SDK.**

[View the Original Repository on GitHub](https://github.com/boto/boto3)

## Key Features

*   **Comprehensive AWS Service Support:** Access and manage a wide range of AWS services, including Amazon S3, Amazon EC2, and many more.
*   **Simplified API:** Write clean, concise, and readable Python code to interact with AWS services.
*   **Object-Oriented Interface:**  Utilize intuitive object-oriented programming for managing AWS resources.
*   **Easy Credential Management:** Configure and manage your AWS credentials securely.
*   **Active Community & Support:** Benefit from a large and active community, along with comprehensive documentation and support resources.

## Getting Started

### Prerequisites

*   A supported version of Python is installed.

### Installation

1.  **Set up a virtual environment** (recommended):

    ```bash
    $ python -m venv .venv
    $ . .venv/bin/activate
    ```

2.  **Install Boto3 using pip:**

    ```bash
    $ python -m pip install boto3
    ```

### Using Boto3

1.  **Configure Credentials:**  Set up your AWS credentials.  You can do this by either setting up the following in a `~/.aws/credentials` file:
   ```ini
   [default]
   aws_access_key_id = YOUR_KEY
   aws_secret_access_key = YOUR_SECRET
   ```
   
    or setting the credentials using environment variables
   
   Other credential configuration methods can be found [here](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html).

2.  **Set a Default Region:**  Configure the default AWS region in a `~/.aws/config` file:

    ```ini
    [default]
    region=us-east-1
    ```

3.  **Import and Use:**  Start interacting with AWS services in your Python code:

    ```python
    import boto3
    s3 = boto3.resource('s3')
    for bucket in s3.buckets.all():
        print(bucket.name)
    ```

## Running Tests

Use `tox` to run tests across all supported Python versions:

```bash
$ tox
```

You can also run tests with `pytest`:

```bash
$ pytest tests/unit
```

## Getting Help

*   **Stack Overflow:**  Ask questions and find answers tagged with `boto3` ([https://stackoverflow.com/questions/tagged/boto3](https://stackoverflow.com/questions/tagged/boto3)).
*   **AWS Support:**  Open a support ticket with [AWS Support](https://console.aws.amazon.com/support/home#/).
*   **GitHub Issues:**  Report bugs and request features on [GitHub](https://github.com/boto/boto3/issues/new).

## Contributing

We welcome contributions!  Please review the `CONTRIBUTING` document ([https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst](https://github.com/boto/boto3/blob/develop/CONTRIBUTING.rst)) before submitting pull requests or issues.

## Maintenance and Support

Boto3 is currently in the full support phase. For more information on SDK version support, see the:

*   [AWS SDKs and Tools Maintenance Policy](https://docs.aws.amazon.com/sdkref/latest/guide/maint-policy.html)
*   [AWS SDKs and Tools Version Support Matrix](https://docs.aws.amazon.com/sdkref/latest/guide/version-support-matrix.html)

## Additional Resources

*   [NOTICE](https://github.com/boto/boto3/blob/develop/NOTICE)
*   [Changelog](https://github.com/boto/boto3/blob/develop/CHANGELOG.rst)
*   [License](https://github.com/boto/boto3/blob/develop/LICENSE)
*   [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)