# Azure SDK for Python: Simplify Cloud Development

Easily build, deploy, and manage applications on Microsoft Azure with the official Azure SDK for Python.  [Get Started with the Azure SDK for Python](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modular Libraries:** Utilize individual, focused libraries for specific Azure services, reducing dependencies and improving code efficiency.
*   **Consistent Design:** Benefit from a consistent API design across all Azure SDKs, simplifying learning and usage.
*   **Production-Ready:** Leverage stable, production-ready libraries for reliable cloud integration.
*   **Latest Features:** Access the latest Azure service capabilities through preview and generally available packages.
*   **Management and Client Libraries:** Access libraries for both provisioning and managing Azure resources and interacting with those resources.

## Getting Started

The Azure SDK for Python offers a suite of libraries, organized by service. To begin, refer to the `README.md` (or `README.rst`) file within the project folder for the specific library you intend to use.

### Prerequisites

The client libraries require Python 3.9 or later. For detailed information on version support, consult the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Service libraries are categorized for ease of use:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries provide access to the latest Azure services and functionalities.  They are announced as **GA** (Generally Available) or in **preview**. They provide common functionalities like retries, logging, and authentication. Review the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/index.html) for more information.

Find the most up-to-date package listings on our [release page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** Use stable, non-preview libraries for production deployments.

### Client: Previous Versions

These are stable, production-ready versions of Azure service libraries. They offer similar functionality to the preview libraries and provide wider service coverage, though they may not fully align with the latest design guidelines.

### Management: New Releases

Management libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available.  These libraries provide core features like intuitive authentication, an HTTP pipeline, error-handling, and distributed tracing.

Documentation and code samples are available [here](https://aka.ms/azsdk/python/mgmt).  A migration guide for transitioning from older libraries is [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the most up-to-date package listings on our [management release page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Note:** For production readiness, use the stable libraries.  If you encounter authentication problems after upgrading, consult the migration guide.

### Management: Previous Versions

For a complete list of management libraries to provision and manage Azure resources, visit the [all packages release page](https://azure.github.io/azure-sdk/releases/latest/all/python.html).  While these may not include all features of the new releases, they offer comprehensive service coverage. Management libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** Comprehensive [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community:** Search existing questions or ask new ones on Stack Overflow using the `azure` and `python` tags.

## Data Collection

The SDK collects information about your usage to improve services. You can opt out of telemetry, as described below. Learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable it, you can pass an instance of `NoUserAgentPolicy` (which is a subclass of `UserAgentPolicy` with an empty `on_request` method) to the client's constructor via the `user_agent_policy` keyword. This example uses the `azure-storage-blob` package.

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

### Reporting Security Issues and Security Bugs

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details.

This project welcomes contributions.  Contributions require a Contributor License Agreement (CLA).  Follow the instructions provided by the CLA-bot when submitting a pull request.

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions or comments.