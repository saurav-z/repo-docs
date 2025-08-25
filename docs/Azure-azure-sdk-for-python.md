# Azure SDK for Python

**Build powerful and reliable applications with the official Azure SDK for Python, offering comprehensive libraries for interacting with Azure services.** [(View the original repository)](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

The Azure SDK for Python provides a comprehensive set of libraries to interact with a wide range of Azure services. Designed for ease of use, reliability, and performance, this SDK empowers developers to build robust cloud-based applications.

## Key Features

*   **Comprehensive Service Coverage:** Access a vast array of Azure services, including compute, storage, networking, databases, and more.
*   **Consistent API Design:** Benefit from a standardized and intuitive API across all service libraries, making it easier to learn and use.
*   **Robust Authentication:** Leverage secure and flexible authentication options, including Azure Active Directory (Azure AD) and shared keys.
*   **Simplified Development:** Get started quickly with clear documentation, sample code, and readily available packages.
*   **Production-Ready Libraries:** Utilize stable, production-ready libraries, as well as preview packages that allow you to explore the latest features.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and improvements through frequent releases.

## Getting Started

To get started with a specific service library, explore the `README.md` (or `README.rst`) file located in the library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For details, review the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

Service libraries are organized into the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries enable you to use and consume existing resources and interact with them, for example: upload a blob. These libraries share several core functionalities such as: retries, logging, transport protocols, authentication protocols, etc. that can be found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. You can learn more about these libraries by reading guidelines that they follow [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** Use stable, non-preview libraries for production.

### Client: Previous Versions

Previous stable versions are available for production usage with Azure. These libraries provide functionality similar to preview versions, but might not fully align with [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) or have the same feature set as newer releases. They offer wider service coverage.

### Management: New Releases

These libraries adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They offer core capabilities shared across all Azure SDKs, like the Azure Identity library, an HTTP Pipeline, error handling, and tracing.
Documentation and code samples are [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [latest packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:** Use stable, non-preview libraries for production. If you experience authentication issues after upgrading, consult the migration guide.

### Management: Previous Versions

For management libraries, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They may have less feature coverage.
Management libraries start with the `azure-mgmt-` namespace (e.g., `azure-mgmt-compute`).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow ([azure](https://stackoverflow.com/questions/tagged/azure) and [python](https://stackoverflow.com/questions/tagged/python) tags)

## Data Collection

The software collects information about your use and sends it to Microsoft.  You can find more information in the help documentation, [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704), and [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default.

To disable, create a `NoUserAgentPolicy` class that subclasses `UserAgentPolicy`, with an empty `on_request` method. Pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` during client creation to disable telemetry.

Example:

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

### Reporting security issues and security bugs

Report privately to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>.  You should get a response within 24 hours. If you don't, follow up via email. Further information, including the MSRC PGP key, is in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).

This project welcomes contributions.  You must agree to a Contributor License Agreement (CLA).
See https://cla.microsoft.com.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.