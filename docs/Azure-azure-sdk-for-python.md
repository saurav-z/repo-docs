# Azure SDK for Python: Simplify Cloud Development

Empower your Python applications with the official Azure SDK, providing comprehensive libraries for interacting with Microsoft Azure services.  Learn more about the Azure SDK for Python by visiting the [original repository](https://github.com/Azure/azure-sdk-for-python).

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modular Libraries:** Utilize individual, focused libraries for each Azure service, reducing dependencies and improving efficiency.
*   **Latest Releases and Previous Versions:** Choose between the newest, actively developed packages (including preview versions) and stable, production-ready releases.
*   **Client and Management Libraries:** Access both client libraries for interacting with resources and management libraries for provisioning and configuration.
*   **Consistent Design Guidelines:** Benefit from libraries that follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a unified development experience.
*   **Authentication and Security:** Leverage secure authentication protocols and reporting for security issues.
*   **Telemetry Configuration:** Offers telemetry collection and opt-out instructions.

## Getting Started

Each service is available through separate libraries.  Find the relevant `README.md` or `README.rst` files located in the library's project folder.

### Prerequisites

The client libraries are supported on Python 3.9 or later. For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

Explore the available packages categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest packages, including those in **GA** (Generally Available) and **preview** states. They provide access to existing resources, with functionalities such as retries, logging, and authentication, shared via the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.  Find the updated list of packages on our [releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE:  For production environments, use the stable, non-preview libraries.

### Client: Previous Versions

Stable, production-ready versions of packages are available for Azure usage.  These offer similar functionalities to the Preview releases, but might not implement all guidelines or have the same feature set.

### Management: New Releases

Leverage management libraries that adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They feature core capabilities, including Azure Identity, an HTTP Pipeline, error handling, and more.  Documentation and samples are available [here](https://aka.ms/azsdk/python/mgmt).  A migration guide for older versions is [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
Find the package list [here](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

> NOTE: For production, choose stable, non-preview libraries. If you encounter authentication issues after upgrading certain packages, consult the migration guide.

### Management: Previous Versions

For a complete list of management libraries, please [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They might not have the same feature set as the new releases but they do offer wider coverage of services.
Management libraries can be identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`

## Need Help?

*   Detailed documentation: [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   File an issue: [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow: Search using `azure` and `python` tags.

## Data Collection

The SDK collects information about your use of the software and sends it to Microsoft. You can turn off telemetry. Learn more in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is on by default.

To opt out, disable telemetry during client construction. Define a `NoUserAgentPolicy` class that is a subclass of `UserAgentPolicy` with an `on_request` method that does nothing. Then pass instance of this class as kwargs `user_agent_policy=NoUserAgentPolicy()` during client creation. This will disable telemetry for all methods in the client. Do this for every new client.

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

### Reporting security issues and security bugs

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. More information is found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

Contribute to this repository, see the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).

This project welcomes contributions. Most contributions require a Contributor License Agreement (CLA).  Details at https://cla.microsoft.com.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.