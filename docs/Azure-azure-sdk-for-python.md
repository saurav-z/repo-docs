# Azure SDK for Python: Build Powerful Python Applications with Azure Services

**Easily build and deploy robust Python applications leveraging the power of Microsoft Azure services with the official Azure SDK for Python.**  ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository hosts the actively developed Azure SDK for Python, providing a comprehensive set of libraries to interact with Azure services.  Explore our public developer docs ([https://docs.microsoft.com/python/azure/](https://docs.microsoft.com/python/azure/)) and versioned docs ([https://azure.github.io/azure-sdk-for-python](https://azure.github.io/azure-sdk-for-python)) for detailed information.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern Design and Guidelines:** Utilize libraries built with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistency and ease of use.
*   **Client and Management Libraries:** Choose between client libraries for interacting with resources and management libraries for provisioning and managing Azure resources.
*   **Production-Ready Libraries:** Benefit from stable, production-ready libraries for reliable application development.
*   **Shared Core Functionality:** Leverage shared features like retries, logging, authentication, and transport protocols through the `azure-core` library.
*   **Azure Identity Library:** Utilize the intuitive Azure Identity library.
*   **HTTP Pipeline:** Access an HTTP pipeline with custom policies, error handling, distributed tracing, and more.
*   **Up-to-Date Releases:** Stay current with the [latest package releases](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

## Getting Started

Each Azure service has its own set of Python libraries. To get started, explore the `README.md` or `README.rst` file in the service-specific project folders located in the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. Refer to our [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for detailed information.

## Packages Available

Azure services offer multiple libraries, organized into these categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries are generally available (GA) or in preview. They enable you to interact with existing resources (e.g., upload a blob). They share core functionalities such as retries, logging, and authentication. Learn more about these libraries by reading the guidelines [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

### Client: Previous Versions

These are the last stable, production-ready versions of the packages. They offer similar functionalities to preview releases but might not follow the same guidelines or have the same features. They may offer a wider service coverage.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide essential capabilities shared among Azure SDKs, including the Azure Identity library, HTTP pipelines, error handling, and distributed tracing.  Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).  A migration guide for transitioning from older versions is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

### Management: Previous Versions

For a complete list of management libraries that enable you to provision and manage Azure resources, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries start with `azure-mgmt-` in their namespace (e.g., `azure-mgmt-compute`).

## Need Help?

*   **Documentation:** Access detailed documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community:** Find answers on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects usage data to improve services. You can disable telemetry collection. See the Microsoft [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for details.

### Telemetry Configuration

Telemetry is enabled by default.  To opt-out, disable telemetry at client construction.  Create a `NoUserAgentPolicy` class that subclasses `UserAgentPolicy` with an empty `on_request` method.  Pass an instance of this class as a `user_agent_policy` kwarg when creating clients:

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

## Reporting Security Issues and Bugs

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  All contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.