# Azure SDK for Python

**Simplify your cloud development with the official Azure SDK for Python, providing comprehensive tools to build, deploy, and manage applications on Microsoft Azure.**

[Link to Original Repo](https://github.com/Azure/azure-sdk-for-python)

## Key Features

*   **Comprehensive Library Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Up-to-date SDK:** Benefit from the latest features and improvements with new releases and updates.
*   **Client and Management Libraries:** Utilize client libraries for interacting with resources (e.g., uploading blobs) and management libraries for provisioning and managing Azure resources.
*   **Consistent Design Guidelines:** The SDK adheres to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for a consistent and intuitive development experience.
*   **Production Ready:** Utilize the stable and non-preview libraries to ensure your code is ready for production.
*   **Simplified Authentication:** Leverage intuitive Azure Identity libraries for secure authentication.
*   **Telemetry Configuration:** Option to disable telemetry collection at client construction using a `NoUserAgentPolicy`.

## Getting Started

Get started with specific libraries by exploring the `README.md` (or `README.rst`) files within each service's project folder in the `/sdk` directory.

### Prerequisites

The client libraries support Python 3.9 and later.

For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The Azure SDK for Python offers libraries categorized as:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These new packages are announced as General Availability (GA) and some are in preview, enabling you to interact with and consume existing Azure resources. They share core functionalities like retries, logging, and authentication protocols. More information can be found in the [Azure SDK Design Guidelines](https://azure.github.io/azure-sdk/python/guidelines/index.html).

Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

>   **Note:** For production use, use stable, non-preview libraries.

### Client: Previous Versions

These are the last stable versions of packages that are production-ready, providing similar functionalities as the preview versions.

### Management: New Releases

New management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) and provide core capabilities such as Azure Identity, HTTP pipelines, error handling, and distributed tracing.

Documentation and code samples can be found [here](https://aka.ms/azsdk/python/mgmt).

A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).

Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

>   **Note:** For production use, use stable, non-preview libraries. Refer to the migration guide if you experience authentication issues after upgrading certain packages.

### Management: Previous Versions

For a complete list of management libraries, [check here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community:** Search or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The SDK collects information about your usage, sent to Microsoft for service improvement. [Privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704). For information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is on by default. To disable, create a `NoUserAgentPolicy` class and pass it during client creation.

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

Report security issues privately to the [Microsoft Security Response Center (MSRC)](mailto:secure@microsoft.com). You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. All contributions require a Contributor License Agreement (CLA).

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).