# Azure SDK for Python: Build Powerful Python Applications with Microsoft Azure

The Azure SDK for Python empowers developers to seamlessly integrate their Python applications with a wide array of Microsoft Azure services.  Explore the official repository: [https://github.com/Azure/azure-sdk-for-python](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Azure Services:** Access a vast collection of Azure services, including storage, compute, networking, databases, and more.
*   **Pythonic Design:** Leverage Pythonic APIs for intuitive and efficient interaction with Azure services.
*   **Modular Libraries:** Choose specific libraries for individual services, promoting code clarity and reducing dependencies.
*   **Active Development:** Benefit from ongoing updates, new features, and improvements.
*   **Cross-Platform Compatibility:** Build and deploy applications across various platforms with ease.

## Getting Started

Get up and running quickly with Azure services. Service libraries reside in the `/sdk` directory. Check the `README.md` (or `README.rst`) in each library's project folder for setup.

### Prerequisites

Ensure your system is running Python 3.9 or later.  See the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Packages Available

Service libraries are organized into the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These new packages are announced as GA and several are currently releasing in preview. They enable you to use and interact with existing resources, like uploading a blob, and share core functionalities found in [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core), such as retries, logging, authentication protocols, etc. Learn more [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the latest package releases [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

These packages offer production-ready, stable versions of the libraries. While they may not implement the guidelines or feature set of the new releases, they offer a broader range of service coverage.

### Management: New Releases

These libraries conform to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide core capabilities, including the Azure Identity library, HTTP Pipeline, error handling, and distributed tracing.

Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt). A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the latest management packages [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:** Use stable, non-preview libraries for production. Refer to the migration guide if you experience authentication issues after upgrading management packages.

### Management: Previous Versions

For the complete list of management libraries, see [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They may have a different feature set than the new releases, but offer broader service coverage. Management libraries have namespaces starting with `azure-mgmt-`, e.g., `azure-mgmt-compute`.

## Need Help?

*   Detailed documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Ask questions: StackOverflow, using `azure` and `python` tags.

## Data Collection

The software collects data about your usage and sends it to Microsoft for service and product improvements.  You can opt-out of telemetry. Learn more in the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is on by default.

To opt out:

1.  Create a `NoUserAgentPolicy` class that subclasses `UserAgentPolicy`.
2.  Define an `on_request` method that does nothing.
3.  When creating a client, pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()`.

Example (using `azure-storage-blob`):

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy


# Create your credential you want to use
mi_credential = ManagedIdentityCredential()

account_url = "https://<storageaccountname>.blob.core.windows.net"

# Set up user-agent override
class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())

container_client = blob_service_client.get_container_client(container=<container_name>)
# TODO: do something with the container client like download blob to a file
```

## Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information is available in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project welcomes contributions.  All contributions require a Contributor License Agreement (CLA).  See https://cla.microsoft.com.

This project adopts the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).