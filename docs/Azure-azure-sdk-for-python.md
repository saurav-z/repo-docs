# Azure SDK for Python: Build Powerful Python Applications with Azure

**Empower your Python applications with seamless integration to Microsoft Azure services using the official Azure SDK for Python.** ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

This repository is the home of the actively developed Azure SDK for Python, providing a comprehensive set of libraries to interact with various Azure services.

## Key Features:

*   **Broad Service Coverage:** Access a wide array of Azure services including compute, storage, networking, databases, and more.
*   **Modern Design:** The SDK adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for consistency and ease of use.
*   **Comprehensive Documentation:** Get started quickly with detailed documentation and code samples available in the [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Client and Management Libraries:** Choose from client libraries for interacting with resources and management libraries for provisioning and managing resources.
*   **Asynchronous Support:** Benefit from asynchronous operations for improved performance and responsiveness.
*   **Authentication:** Utilize the intuitive Azure Identity library for secure and streamlined authentication.
*   **Shared Core Functionality:** Leverage shared capabilities like retries, logging, authentication, and transport protocols provided in the `azure-core` library.
*   **Telemetry and Privacy:** Control data collection with clear guidance on [Telemetry Configuration](https://github.com/Azure/azure-sdk-for-python#telemetry-configuration) to ensure privacy.
*   **Active Development:** Benefit from the latest features and improvements as the SDK is under constant development.

## Getting Started

Each Azure service has its own dedicated library. To get started, find the `README.md` (or `README.rst`) file within the specific library's project folder in the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later. Refer to the [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for more details.

## Packages Available

The SDK offers various libraries categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries are the latest releases, including GA (Generally Available) and preview versions. These enable you to use and consume Azure resources and interact with them (e.g., uploading a blob). Find the most up-to-date list on our [releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

> **Note:** For production use, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions of packages, production-ready and offering similar functionalities to the preview ones. Although they may not implement the [guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html) or have the same feature set as the latest releases, they offer wider service coverage.

### Management: New Releases

Management libraries following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are now available. These libraries offer core capabilities shared across all Azure SDKs, including the Azure Identity library and error-handling. Documentation and code samples are available [here](https://aka.ms/azsdk/python/mgmt). A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Note:** If you are experiencing authentication issues with the management libraries after upgrading certain packages, please refer to the migration guide mentioned above for proper instructions.

### Management: Previous Versions

For a list of management libraries for provisioning and managing Azure resources, [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). These libraries are identified by namespaces that start with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow (use tags `azure` and `python`)

## Data Collection

The software collects information that may be sent to Microsoft. You can disable telemetry by modifying your code as detailed below. For more information, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry collection is on by default.  To disable telemetry, create a `NoUserAgentPolicy` as described in the original README and apply it during client construction.

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

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. All contributions require a Contributor License Agreement (CLA). This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).