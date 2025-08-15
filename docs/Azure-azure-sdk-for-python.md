# Azure SDK for Python: Simplify Cloud Development with Powerful Python Libraries

**Build robust and scalable applications on Azure with the official Azure SDK for Python.**

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository hosts the active development of the Azure SDK for Python, providing a comprehensive suite of libraries to interact with Azure services.  For comprehensive documentation and getting started guides, explore the [public developer docs](https://docs.microsoft.com/python/azure/) or our versioned [developer docs](https://azure.github.io/azure-sdk-for-python).  **[See the original repository here](https://github.com/Azure/azure-sdk-for-python)**.

## Key Features

*   **Modular Libraries:** Utilize service-specific libraries for a streamlined development experience.
*   **Comprehensive Service Coverage:** Access a wide range of Azure services with dedicated Python libraries.
*   **Client and Management Libraries:** Leverage both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Consistent Design:** Follows the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a unified and intuitive developer experience.
*   **Core Functionality:** Benefit from shared core features like retries, logging, authentication, and more via the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Production Ready:** Choose from stable, non-preview libraries for production use.
*   **Cross-Platform Support:** Compatible with Python 3.9 and later.

## Getting Started

*   **Install Specific Libraries:** Instead of a single large package, each service has its own library. Find the `README.md` (or `README.rst`) file within the library's project folder to start.
*   **Locate Service Libraries:** Service libraries can be found within the `/sdk` directory.
*   **Version Support:** Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details on supported Python versions.

## Package Categories

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries offer GA (Generally Available) and preview releases for consuming existing resources. These libraries share core functionalities like retries, logging, transport protocols, and authentication protocols, as found in the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
Read more about these libraries via the guidelines they follow [here](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> NOTE: Production-ready libraries are generally stable, non-preview releases.

### Client: Previous Versions

These libraries offer the last stable versions of packages for use with Azure and are production-ready. They offer similar functionality to the preview releases, allowing you to use and consume existing resources. Note that they may not have the same feature set or adhere to the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) as the newer releases, but offer wider service coverage.

### Management: New Releases

These new management libraries adhere to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). They provide core capabilities such as the intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and more.
Documentation and code samples for these new libraries can be found [here](https://aka.ms/azsdk/python/mgmt).
A migration guide for transitioning from older versions is located [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> NOTE: For production use, use a stable, non-preview library.  If you're experiencing authentication issues after upgrading management libraries, consult the migration guide.

### Management: Previous Versions

For a complete list of management libraries enabling you to provision and manage Azure resources, check [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). They may not have all the features of the new releases but offer more service coverage. These libraries are identified by namespaces starting with `azure-mgmt-`, for example, `azure-mgmt-compute`.

## Need Help?

*   **Comprehensive Documentation:** Access detailed documentation at [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **GitHub Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Stack Overflow:** Search [previous questions](https://stackoverflow.com/questions/tagged/azure+python) or ask new ones using the tags `azure` and `python`.

## Data Collection

This software collects information about you and your usage of the software, which is sent to Microsoft. Microsoft may use this information to provide services and improve products. You can turn off telemetry, as described below.
Learn more in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).
For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry collection is on by default.

To opt-out, disable telemetry at client construction.  Define a `NoUserAgentPolicy` class that subclasses `UserAgentPolicy` and has an `on_request` method that does nothing.  Then pass an instance of this class with `user_agent_policy=NoUserAgentPolicy()` during client creation.  Do this for every new client.

Here's an example using `azure-storage-blob`:

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

Security issues and bugs should be reported privately, via email, to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>. You should receive a response within 24 hours. If for some reason you do not, please follow up via email to ensure we received your original message. Further information, including the MSRC PGP key, can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

Refer to the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details.

This project welcomes contributions, which require you to agree to a Contributor License Agreement (CLA). Learn more at https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will determine if you need a CLA and guide you.

This project adheres to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information, see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.