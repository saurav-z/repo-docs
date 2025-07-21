# Azure SDK for Python: Build Powerful Python Applications for the Cloud

**Empower your Python projects with the Azure SDK, providing comprehensive libraries for interacting with Azure services and building robust, scalable applications.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

The Azure SDK for Python offers a rich set of libraries, designed to simplify your development process. Visit the [official GitHub repository](https://github.com/Azure/azure-sdk-for-python) for the source code and more information.

## Key Features

*   **Comprehensive Service Coverage:**  Libraries for a wide range of Azure services including compute, storage, networking, databases, and more.
*   **Production-Ready Libraries:** Stable, well-documented libraries suitable for production environments, along with preview releases for early access.
*   **Consistent Design:** Libraries follow the Azure SDK Design Guidelines for Python, ensuring a cohesive and intuitive developer experience.
*   **Authentication and Authorization:** Seamless integration with Azure Active Directory and other authentication methods.
*   **Error Handling and Retries:** Built-in error handling and retry mechanisms to improve application resilience.
*   **Modern SDK Design:**  Includes features like HTTP Pipeline with custom policies, error-handling, distributed tracing, and more.

## Getting Started

Each Azure service has its own dedicated library.  To begin, consult the `README.md` (or `README.rst`) file within the respective library's project folder in the `/sdk` directory.

### Prerequisites

The client libraries require Python 3.9 or later. For the latest version support policy, please read our [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

The SDK packages are categorized for easier access:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** (Generally Available) and **preview** packages providing access to existing resources.  They offer features like retries, logging, transport, and authentication. The core functionality can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Review the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) these libraries follow.

Find the newest packages on our [releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production environments, use stable, non-preview libraries.

### Client: Previous Versions

These are the stable, production-ready versions of packages, offering similar functionalities to the preview releases. They may not align perfectly with the latest guidelines but provide broader service coverage.

### Management: New Releases

Management libraries built using the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available, including features like Azure Identity, an HTTP Pipeline, and more. Documentation is [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

See the [latest management packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** Production applications should use stable, non-preview libraries. Migration from older packages might require adjustments to your authentication code; refer to the migration guide.

### Management: Previous Versions

For a complete list of management libraries for managing Azure resources, see the [complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow (use tags: `azure` and `python`)

## Data Collection

The software gathers data about your usage and may send it to Microsoft for service improvements. You can disable telemetry. Learn more in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more info on the Azure SDK's collected data, visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable telemetry, override the `UserAgentPolicy` class. See the example below.

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

### Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. More info can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.

This project welcomes contributions.  All contributions require a Contributor License Agreement (CLA). Visit https://cla.microsoft.com for details.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.