# Azure SDK for Python: Build Powerful Python Applications with Azure (SEO Optimized)

**Unlock the power of Microsoft Azure with the Azure SDK for Python, a comprehensive library that simplifies building, deploying, and managing applications on the Azure cloud.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the foundation for the Azure SDK for Python, offering developers a streamlined and efficient way to interact with Azure services. For detailed documentation and examples, please visit the [public developer docs](https://docs.microsoft.com/python/azure/) or the versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

## Key Features of the Azure SDK for Python

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Simplified Development:** The SDK provides intuitive APIs and client libraries, making it easy to integrate Azure services into your Python applications.
*   **Cross-Platform Compatibility:** Supports Python 3.9 and later, enabling development on various operating systems.
*   **Modular Design:** Each service has its own dedicated library, allowing you to use only the components you need, reducing project size and dependencies.
*   **Azure SDK Design Guidelines:** Follows the established [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistent and reliable performance.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and improvements through frequent releases and package updates.
*   **Authentication Simplified:** Integrated with the intuitive [Azure Identity library](https://pypi.org/project/azure-identity/) for secure access to Azure resources.
*   **Enhanced Management Capabilities:** Provides management libraries for provisioning and managing Azure resources.
*   **Telemetry and Data Collection:** Includes information on data collection practices and how to disable telemetry for privacy.

## Getting Started

Each Azure service has a dedicated set of libraries. To get started, find the `README.md` (or `README.rst`) file within the respective library's project folder in the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

The SDK libraries are organized into different categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries are newly released and generally available (GA), with some in preview. They provide access to and allow you to interact with existing Azure resources (e.g., uploading a blob). They share core functionalities from the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library, such as retries, logging, and authentication.  Find the [most up-to-date list of new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE: For production use, utilize the stable, non-preview libraries.

### Client: Previous Versions

These are stable, production-ready versions of packages. While they provide similar functionality to the Preview versions, they may have a different feature set and might not align with the latest guidelines.  They offer wider coverage of services.

### Management: New Releases

These new management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide core capabilities such as the Azure Identity library, an HTTP Pipeline, error handling, and distributed tracing. Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt). A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up to date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> NOTE: If you need to ensure your code is ready for production use, use one of the stable, non-preview libraries. Check the migration guide if upgrading packages.

### Management: Previous Versions

For a complete list of management libraries that enable you to provision and manage Azure resources, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`

## Need Help?

*   Detailed documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Ask questions on Stack Overflow: use the `azure` and `python` tags.
*   Check [previous questions](https://stackoverflow.com/questions/tagged/azure+python)

## Data Collection

The software may collect information about your use of the software and send it to Microsoft. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is enabled by default.

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

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project welcomes contributions. Most contributions require you to agree to a Contributor License Agreement (CLA). Visit https://cla.microsoft.com for details.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.