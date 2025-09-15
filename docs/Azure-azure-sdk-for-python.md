# Azure SDK for Python: Simplify Cloud Development

**The Azure SDK for Python empowers developers to seamlessly build and deploy applications on Microsoft Azure, offering robust tools and libraries for a variety of services.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services with dedicated Python libraries.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with resources and management libraries for provisioning and managing Azure resources.
*   **Consistent Design & Guidelines:** Leverage libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a consistent and intuitive development experience.
*   **Core Functionality:** Benefit from shared core features like retries, logging, authentication, and transport protocols via the `azure-core` library.
*   **Version Support:**  Supports Python 3.9 and later.
*   **Telemetry Opt-Out:** Ability to disable telemetry collection for privacy.

## Getting Started

Each Azure service has a dedicated library.  To get started:

1.  Explore the `/sdk` directory to locate service libraries.
2.  Refer to the `README.md` (or `README.rst`) file within each library's project folder for service-specific instructions.

## Available Packages

The Azure SDK for Python offers a range of packages categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest Generally Available (GA) and preview packages.  They allow you to utilize existing resources, such as uploading blobs.  Ensure your code is production-ready by using the stable, non-preview libraries.

[Find the most up-to-date list of packages here](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

### Client: Previous Versions

These are the stable, production-ready, previous versions of client libraries. They offer similar functionality to the new releases, but may not have all the same features or follow the latest guidelines.

### Management: New Releases

These libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide core capabilities shared across all Azure SDKs.

[Find the most up-to-date list of packages here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

>   **Note:** Ensure that you are using stable libraries for production use.  If you are upgrading, please refer to the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for proper authentication code updates.

### Management: Previous Versions

These libraries are for provisioning and managing Azure resources.

[Check here for the complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html)

Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** StackOverflow ([azure](https://stackoverflow.com/questions/tagged/azure) and [python](https://stackoverflow.com/questions/tagged/python) tags)

## Data Collection

The Azure SDK collects data to improve its services, as described in Microsoft's [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For more information on the data collected, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable:
1.  Define a `NoUserAgentPolicy` class as a subclass of `UserAgentPolicy` with an `on_request` method that does nothing.
2.  Pass an instance of this class as `user_agent_policy=NoUserAgentPolicy()` during client creation for each new client.

Example:

```python
import os
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobServiceClient
from azure.core.pipeline.policies import UserAgentPolicy

# Create your credential
mi_credential = ManagedIdentityCredential()
account_url = "https://<storageaccountname>.blob.core.windows.net"

# Set up user-agent override
class NoUserAgentPolicy(UserAgentPolicy):
    def on_request(self, request):
        pass

# Create the BlobServiceClient object
blob_service_client = BlobServiceClient(account_url, credential=mi_credential, user_agent_policy=NoUserAgentPolicy())
container_client = blob_service_client.get_container_client(container=<container_name>)
```

## Reporting Security Issues

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for more information.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).