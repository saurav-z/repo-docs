# Azure SDK for Python: Simplify Cloud Development

**Empower your Python applications with seamless integration and robust functionality using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository is the home for the active development of the Azure SDK for Python. It provides Python developers with libraries that offer native support for Azure services.  For detailed documentation and examples, please visit the [public developer docs](https://docs.microsoft.com/python/azure/) or our versioned [developer docs](https://azure.github.io/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern SDK Design:** Leverage libraries built with the latest [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a consistent and intuitive development experience.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure infrastructure.
*   **Simplified Authentication:** Easily integrate with Azure Active Directory and other authentication methods.
*   **Production-Ready:** Benefit from stable, tested, and production-ready libraries for reliable performance.
*   **Community-Driven:** Contribute to the open-source project and collaborate with other developers.

## Getting Started

Each Azure service has a dedicated set of libraries, allowing you to select the specific packages you need.  Refer to the `README.md` or `README.rst` file within each library's project folder for detailed instructions.  Libraries are located in the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Available Packages

The Azure SDK for Python organizes libraries into categories based on their function and release status:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest, GA (Generally Available) and preview packages for interacting with Azure resources. These libraries provide a consistent set of core functionalities such as retries, logging, and authentication, leveraging the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library. Find the [latest package list here](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Important:** For production, choose stable, non-preview libraries.

### Client: Previous Versions

Stable, production-ready libraries providing access to Azure services. These may not implement all the latest guidelines or have the same feature set as newer releases, but they may offer wider service coverage.

### Management: New Releases

These libraries adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide capabilities like Azure Identity, HTTP pipeline features, error handling, and distributed tracing.  Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is available to help transition from older versions: [Migration Guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). The [most up-to-date list of packages is here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> **Important:** Use stable, non-preview libraries in production. Review the migration guide if you're upgrading to the new management SDKs.

### Management: Previous Versions

Libraries for provisioning and managing Azure resources.  For a complete list, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries offer a wider coverage of services.  Look for namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** [StackOverflow](https://stackoverflow.com/questions/tagged/azure+python) with tags `azure` and `python`.

## Data Collection

The Azure SDK for Python collects telemetry data to improve products and services. You can control telemetry as described below. Learn more in the help documentation and the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704) and the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default.

To opt out, disable telemetry during client construction. Create a custom `NoUserAgentPolicy` class by subclassing `UserAgentPolicy` and overriding the `on_request` method to do nothing. Then, pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` when creating your client. This disables telemetry for all methods in the client. Repeat for each new client.

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

Report security vulnerabilities privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

Contribute to the project by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).