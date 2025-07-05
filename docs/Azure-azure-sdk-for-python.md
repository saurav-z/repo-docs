# Azure SDK for Python: Build Powerful Python Applications with Azure

**Empower your Python projects with the Azure SDK, offering a comprehensive suite of libraries to seamlessly integrate with Microsoft Azure services.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the actively developed Azure SDK for Python, with libraries designed for easy integration with various Azure services. Explore our public developer docs ([https://docs.microsoft.com/python/azure/](https://docs.microsoft.com/python/azure/)) and versioned developer docs ([https://azure.github.io/azure-sdk-for-python](https://azure.github.io/azure-sdk-for-python)) for in-depth information.

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services through dedicated Python libraries.
*   **Modular Design:** Utilize individual libraries for specific Azure services instead of a single, large package.
*   **Client and Management Libraries:** Access both new (GA and preview) and previous versions for flexibility.
*   **Shared Core Functionality:** Benefit from shared features like retries, logging, authentication, and more, provided by the `azure-core` library.
*   **Azure SDK Design Guidelines:** Management libraries adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).

## Getting Started

1.  **Choose your library:** Select the specific Azure service library you need from the `/sdk` directory.
2.  **Review the README:** Each service library's `README.md` (or `README.rst`) file provides detailed instructions.
3.  **Prerequisites:**  Ensure you have Python 3.9 or later. See the [version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Available Packages

Choose from the following categories of libraries:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are General Availability (GA) and preview packages that allow you to use and consume existing Azure resources. They share core functionalities such as: retries, logging, transport protocols, authentication protocols. Find the most up-to-date list of all new packages [here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

**Note:** For production use, consider stable, non-preview libraries.

### Client: Previous Versions

These packages are the last stable versions of production-ready Azure libraries, providing similar functionalities to the preview versions but potentially with wider service coverage.

### Management: New Releases

Leveraging the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), these libraries provide core capabilities such as the Azure Identity library, an HTTP Pipeline, error-handling, and distributed tracing.  Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the latest management package releases [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

**Note:**  Use stable, non-preview libraries for production environments.  Refer to the migration guide if you encounter authentication issues after upgrading management packages.

### Management: Previous Versions

Access a complete list of management libraries for provisioning and managing Azure resources [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries, identified by `azure-mgmt-` namespaces, may offer wider service coverage.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:**  Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection and Telemetry

The software collects data to improve services. Review the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) for details.

### Telemetry Configuration

Telemetry is enabled by default. You can opt-out by disabling telemetry at client construction:

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.  Find further information in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.  All contributions require a Contributor License Agreement (CLA).  Learn more at https://cla.microsoft.com.

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.