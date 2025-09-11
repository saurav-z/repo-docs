# Azure SDK for Python: Build Powerful Python Applications on Azure

**Empower your Python projects with the official Azure SDK for Python, offering comprehensive tools for interacting with Azure services.**  ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

This repository is the central hub for the active development of the Azure SDK for Python, providing developers with the latest libraries and tools to integrate Azure services into their Python applications.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern Design:** The SDK adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), ensuring consistency and ease of use.
*   **Client and Management Libraries:** Leverage client libraries for direct interaction with Azure resources and management libraries for provisioning and managing Azure infrastructure.
*   **Production-Ready and Preview Packages:** Choose from stable, production-ready libraries or explore the latest preview packages for early access to new features.
*   **Core Functionality:** Benefit from shared core functionalities like retries, logging, authentication, and transport protocols provided by the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Detailed Documentation:** Access comprehensive documentation, including developer guides, code samples, and migration guides, to help you get started quickly.
*   **Robust Ecosystem:** Benefit from a large and active community, with support through GitHub Issues and Stack Overflow.

## Getting Started

Each Azure service is available through a separate library. To get started, find the `README.md` (or `README.rst`) file in the specific library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. Review the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

Libraries are categorized into the following:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These packages are **GA** (General Availability) or in **preview**, enabling interaction with existing resources, like uploading a blob. They share core functionalities (retries, logging, etc.) from the `azure-core` library. Learn more in the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [most up-to-date package list](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production use, opt for stable, non-preview libraries.

### Client: Previous Versions

These are stable, production-ready versions of packages. They provide similar functionalities but may not implement all guidelines or have the same features as newer releases. They may offer wider service coverage.

### Management: New Releases

Use the latest management libraries, following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide core features like Azure Identity, an HTTP Pipeline, error handling, and tracing.

Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt) and a migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up-to-date package list](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** Ensure you are using a stable library for production.  If you experience authentication issues after upgrading, see the migration guide.

### Management: Previous Versions

Find a list of libraries to manage and provision Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries use the `azure-mgmt-` namespace (e.g., `azure-mgmt-compute`).

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search with `azure` and `python` tags.

## Data Collection and Telemetry

This software collects data that may be sent to Microsoft. You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For details on the data collected by the Azure SDK, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default.  To disable:

1.  Define a `NoUserAgentPolicy` (subclass of `UserAgentPolicy`) with an empty `on_request` method.
2.  Pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` during client creation.

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

## Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.  See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.