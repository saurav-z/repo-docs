# Azure SDK for Python: Build Powerful Python Applications on Azure

**Empower your Python projects with seamless integration to Azure services using the official Azure SDK for Python.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html)
[![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html)
[![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html)
[![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

This repository provides the source code and resources for developing and utilizing the Azure SDK for Python, offering a comprehensive set of libraries to interact with various Azure services.  Find documentation and examples on the [public developer docs](https://docs.microsoft.com/python/azure/) and [versioned developer docs](https://azure.github.io/azure-sdk-for-python).

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, from storage and compute to databases and AI, all within your Python applications.
*   **Modular Library Design:** Choose specific libraries for each Azure service, allowing you to include only the dependencies you need for a streamlined development experience.
*   **Robust Core Functionality:** Benefit from shared features like retries, logging, authentication, and transport protocols provided by libraries such as `azure-core`.
*   **Management Libraries:** Provision and manage Azure resources using management libraries, leveraging intuitive Azure Identity and other core capabilities.
*   **Regular Updates:** Stay up-to-date with the latest features and improvements through ongoing development and releases.

## Getting Started

To get started with a specific service library, see the `README.md` (or `README.rst`) file located in the library's project folder within the `/sdk` directory.

### Prerequisites

The client libraries are supported on Python 3.9 or later.  For more details, please read our page on [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy).

## Packages Available

The SDK provides libraries categorized by service type and release status. Each service may have multiple libraries available:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These libraries provide GA (Generally Available) and preview access to use and consume existing Azure resources.  They offer core functionalities found in the `azure-core` library.  Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

**Important Note:** For production use, utilize stable, non-preview libraries.

### Client: Previous Versions

Stable, production-ready versions of client libraries provide similar functionalities to the preview versions. They offer wider service coverage, though may not always implement the latest guidelines.

### Management: New Releases

Management libraries adhering to the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) are available.  These libraries provide core capabilities, including Azure Identity and HTTP pipeline features. Find documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).  A migration guide is available [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide). Find the [most up to date list of all of the new packages on our page](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

**Important Note:**  For production, use stable libraries. If experiencing authentication issues after upgrading, consult the migration guide.

### Management: Previous Versions

These libraries enable you to provision and manage Azure resources.  Find a complete list [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html). Management libraries are identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   **Comprehensive Documentation:**  [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **Report Issues:** [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Community Support:**  Search or ask questions on StackOverflow using the tags `azure` and `python`.

## Data Collection & Telemetry

The SDK collects usage data to improve services.

### Telemetry Configuration

Telemetry collection is enabled by default.

To disable telemetry:

1.  Define a `NoUserAgentPolicy` class that subclasses `UserAgentPolicy` with an `on_request` method that does nothing.
2.  Pass an instance of `NoUserAgentPolicy` as the `user_agent_policy` in the client constructor:

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

Report security issues and bugs privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>. You should receive a response within 24 hours.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. This project welcomes contributions; all contributions require a Contributor License Agreement (CLA).

*   [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
*   Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.