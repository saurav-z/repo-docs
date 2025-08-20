# Azure SDK for Python: Build Powerful Python Applications on Azure

**Easily access and manage Azure services with the official Azure SDK for Python, enabling developers to build robust and scalable applications.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the latest Python client libraries for interacting with Azure services, enabling you to build, deploy, and manage applications on the Microsoft Azure cloud platform. Explore a range of SDKs to streamline development and leverage the full power of Azure.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide array of Azure services, including compute, storage, networking, databases, and more.
*   **Modern Design:**  Leverage libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring a consistent and intuitive developer experience.
*   **Asynchronous Support:**  Build high-performance, scalable applications using asynchronous programming capabilities.
*   **Authentication:** Seamlessly integrate with various Azure authentication methods, including Azure Active Directory (Azure AD) and managed identities.
*   **Production-Ready Libraries:** Choose from stable, non-preview libraries for production environments, ensuring reliability and support.
*   **Management Libraries:** Provision and manage Azure resources with the comprehensive management libraries.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and improvements through frequent library updates.

## Getting Started

To begin, choose the specific service library you need and refer to its `README.md` (or `README.rst`) file within the `/sdk` directory for detailed instructions.

### Prerequisites

The client libraries require Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

The Azure SDK for Python provides libraries categorized as follows:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries offer access to existing Azure resources, allowing you to interact with them, and provide core functionalities such as retries, logging, and authentication. Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** Use stable, non-preview libraries for production use.

### Client: Previous Versions

These are the last stable versions of packages for production usage. They offer similar functionalities to the Preview releases but with a wider coverage of services.

### Management: New Releases

New management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and offer shared capabilities like the Azure Identity library and error handling.  Documentation and samples are available [here](https://aka.ms/azsdk/python/mgmt). Refer to the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for upgrading from older versions. Find the [latest management packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:**  If you experience authentication issues after upgrading management packages, review the migration guide.

### Management: Previous Versions

For a complete list of management libraries to provision and manage Azure resources, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).  Management libraries are identified by namespaces starting with `azure-mgmt-`, for example, `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow with `azure` and `python` tags.

## Data Collection

This software collects data to improve services.  You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is on by default.

To opt out, you can disable telemetry at client construction. Define a `NoUserAgentPolicy` class that is a subclass of `UserAgentPolicy` with an `on_request` method that does nothing. Then pass instance of this class as kwargs `user_agent_policy=NoUserAgentPolicy()` during client creation. This will disable telemetry for all methods in the client. Do this for every new client.

The example below uses the `azure-storage-blob` package. In your code, you can replace `azure-storage-blob` with the package you are using.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.  See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing. All contributions require a Contributor License Agreement (CLA).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.