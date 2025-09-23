# Azure SDK for Python: Simplify Your Cloud Development

**Harness the power of Azure with the official Python SDK, offering a comprehensive set of libraries to build and manage your cloud applications.** [(View the original repository)](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the source code and resources for the active development of the Azure SDK for Python.  Access comprehensive documentation and examples for developers: [public developer docs](https://docs.microsoft.com/python/azure/) and [versioned developer docs](https://azure.github.io/azure-sdk-for-python).

## Key Features

*   **Comprehensive Library Coverage:** Access a wide range of libraries for interacting with various Azure services.
*   **Modular Design:** Utilize individual, focused libraries for specific Azure services, promoting efficient development.
*   **Client and Management Libraries:** Choose between client libraries for interacting with resources and management libraries for provisioning and managing resources.
*   **Consistent Design Guidelines:** Benefit from libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) offering shared core capabilities.
*   **Production-Ready and Preview Packages:** Access both stable, production-ready libraries and preview releases to stay ahead of the curve.
*   **Up-to-date Package Listings:** Stay current with the latest package releases and versions.

## Getting Started

To begin using a specific Azure service, locate the relevant library within the `/sdk` directory and refer to its `README.md` or `README.rst` file for detailed instructions.

### Prerequisites

Ensure your environment meets the following requirements:

*   Python 3.9 or later.  For more details, refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Explore the available libraries organized by service category:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

Leverage the latest GA (Generally Available) and preview client libraries to consume and interact with existing Azure resources, such as uploading blobs. These libraries share core features like retries, logging, authentication, and more, leveraging the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Discover [the guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) that govern these libraries.

Find the [most up-to-date list of packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Important:** For production use, opt for stable, non-preview libraries.

### Client: Previous Versions

Access stable, production-ready versions of packages for Azure service interaction. These provide functionalities to use and consume existing resources. Note they may not implement the latest [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) or have the same feature set as the latest releases, but offer broader service coverage.

### Management: New Releases

Use the new management libraries, aligned with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), for provisioning and managing Azure resources. Benefit from core capabilities like Azure Identity, HTTP Pipeline, error-handling, and distributed tracing.

*   Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).
*   Consult the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for transitioning from older library versions.

Find the [latest packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Important:** For production use, prioritize stable libraries. If experiencing authentication issues after upgrading management libraries, review the migration guide.

### Management: Previous Versions

Explore the complete list of management libraries to provision and manage Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries, identified by `azure-mgmt-` namespaces (e.g., `azure-mgmt-compute`), offer broader service coverage, though they might not include the complete feature set of the newest releases.

## Need Help?

*   For detailed documentation, visit our [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   File an issue via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   Search existing questions or ask new ones on Stack Overflow using the tags `azure` and `python`.

## Data Collection

The software collects information about your use of the software and sends it to Microsoft to provide services and improve products and services.
You may turn off telemetry as described below. Learn more about data collection and use in the help documentation and Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).
For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is enabled by default.

To opt out, disable telemetry during client creation.  Subclass `UserAgentPolicy` with a `NoUserAgentPolicy` and pass an instance of this class as `user_agent_policy` during client construction.

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

## Reporting Security Issues and Bugs

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. If you don't, follow up via email. Find further information, including the MSRC PGP key, in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

For details on contributing, consult the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).

This project welcomes contributions. All contributors must agree to a Contributor License Agreement (CLA). Visit https://cla.microsoft.com for details.

When you submit a pull request, a CLA-bot will guide you.

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.