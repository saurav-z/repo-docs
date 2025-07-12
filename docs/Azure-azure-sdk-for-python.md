# Azure SDK for Python: Build Powerful Python Applications on Azure

**Enhance your Python projects with the official Azure SDK, offering a comprehensive suite of libraries to seamlessly integrate with Azure services.** [(View the original repository)](https://github.com/Azure/azure-sdk-for-python)

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, and more.
*   **Easy Integration:** Simplify interactions with Azure services using intuitive and Pythonic APIs.
*   **Production-Ready Libraries:** Choose stable, production-ready libraries for reliable performance.
*   **Modern Design:** Benefit from libraries built following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Flexible Authentication:** Leverage various authentication methods for secure access to your Azure resources.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and improvements.
*   **Telemetry and Control:** Customize your Telemetry with built-in control options.

## Getting Started

The Azure SDK for Python provides individual libraries for specific Azure services, allowing you to include only the dependencies your project requires.

*   Consult the `README.md` (or `README.rst`) file within each service's project folder for library-specific instructions.
*   Find service libraries within the `/sdk` directory.

### Prerequisites

*   Supported on Python 3.9 or later.
*   Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.

## Available Packages

Azure SDK packages are categorized into:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

*   **GA (General Availability) and Preview Releases:** Utilize the latest packages for existing resources and interact with them (e.g., uploading blobs).
*   **Shared Core Functionality:** Benefit from shared features like retries, logging, transport protocols, and authentication, provided by the [azure-core](https://github.com/Azure/azure-sdk-for-python/blob/main/sdk/core/azure-core) library.
*   **Guidelines:** These libraries follow established [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) for consistency.
*   **Up-to-date List:** Find the [most up-to-date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

>   **Important:** Use stable, non-preview libraries for production environments.

### Client: Previous Versions

*   **Stable Versions:** Leverage the last stable, production-ready versions of Azure packages.
*   **Similar Functionality:** These offer similar functionalities to preview versions, but may not fully align with the latest [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html).
*   **Wider Service Coverage:** These may offer coverage for a wider range of services.

### Management: New Releases

*   **Azure SDK Design Guidelines:** Take advantage of a new set of management libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/).
*   **Core Capabilities:** Access core features shared across Azure SDKs, including Azure Identity, HTTP Pipeline, error handling, and distributed tracing.
*   **Documentation and Samples:** Access documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).
*   **Migration Guide:** Consult the [migration guide](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide) for transitioning from older versions.
*   **Up-to-date List:** Find the [most up to date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

>   **Important:** Use stable, non-preview libraries for production. Review the migration guide if you encounter authentication issues after upgrading management libraries.

### Management: Previous Versions

*   **Complete List:** Find a complete list of management libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).
*   **Wider Coverage:** While these might not include all the features of newer releases, they often provide broader service support.
*   **Identification:** Management libraries are identifiable by namespaces starting with `azure-mgmt-`, such as `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** Detailed documentation is available at [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects data about your use of the software and sends it to Microsoft. Microsoft may use this information to provide services and improve our products and services. You may turn off the telemetry.
*   You can learn more about data collection and use in the help documentation and Microsoft’s [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).
*   For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default.

To disable telemetry:
1.  Create a custom `NoUserAgentPolicy` class that subclasses `UserAgentPolicy` and overrides the `on_request` method to do nothing.
2.  Pass an instance of this class as the `user_agent_policy` argument during client creation (e.g., `user_agent_policy=NoUserAgentPolicy()`). Do this for every new client.

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

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours. Further information, including the MSRC PGP key, can be found in the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue).

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on contributing.

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA). Visit https://cla.microsoft.com for details.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.