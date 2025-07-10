# Azure SDK for Python: Simplify Cloud Development 

**Empower your Python applications with the Azure SDK for Python, offering a comprehensive suite of libraries to seamlessly interact with Azure services.**  ([Original Repository](https://github.com/Azure/azure-sdk-for-python))

[![Packages](https://img.shields.io/badge/packages-latest-blue.svg)](https://azure.github.io/azure-sdk/releases/latest/python.html) [![Dependencies](https://img.shields.io/badge/dependency-report-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencies.html) [![DepGraph](https://img.shields.io/badge/dependency-graph-blue.svg)](https://azuresdkartifacts.blob.core.windows.net/azure-sdk-for-python/dependencies/dependencyGraph/index.html) [![Python](https://img.shields.io/pypi/pyversions/azure-core.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure/) [![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/python/python%20-%20core%20-%20ci?branchName=main)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=458&branchName=main)

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide array of Azure services, including compute, storage, networking, databases, and more.
*   **Modern Design:**  Leverages the latest Azure SDK design guidelines for a consistent and intuitive developer experience.
*   **Flexible Package Structure:** Utilize individual, focused libraries for each Azure service, allowing you to minimize dependencies and optimize your application's footprint.
*   **Client and Management Libraries:** Choose between client libraries for interacting with existing resources (e.g., uploading a blob) and management libraries for provisioning and managing Azure resources.
*   **Shared Core Functionality:** Benefit from core features like retries, logging, authentication, and transport protocols, available through the `azure-core` library.
*   **Well-Defined Guidelines:**  Libraries follow guidelines for consistency and ease of use ([https://azure.github.io/azure-sdk/python/guidelines/index.html](https://azure.github.io/azure-sdk/python/guidelines/index.html))
*   **Detailed Documentation:** Access comprehensive documentation to get started quickly and understand each library's functionality ([https://aka.ms/python-docs](https://aka.ms/python-docs)).
*   **Version Support Policy:**  Supported on Python 3.9 or later (see [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy)).
*   **Active Development:** Stay up-to-date with the latest features and improvements through ongoing development and releases.

## Getting Started

Each Azure service has a dedicated Python library. To start, review the `README.md` (or `README.rst`) file within the corresponding library's project folder located in the `/sdk` directory.

## Available Packages

The Azure SDK for Python offers libraries categorized as follows:

*   **Client: New Releases** - GA and preview libraries for interacting with existing resources. ([https://azure.github.io/azure-sdk/releases/latest/index.html#python](https://azure.github.io/azure-sdk/releases/latest/index.html#python))
*   **Client: Previous Versions** - Stable, production-ready versions of client libraries.
*   **Management: New Releases** - New management libraries adhering to Azure SDK design guidelines ([https://aka.ms/azsdk/python/mgmt](https://aka.ms/azsdk/python/mgmt)).
*   **Management: Previous Versions** - Libraries for provisioning and managing Azure resources.  ([https://azure.github.io/azure-sdk/releases/latest/all/python.html](https://azure.github.io/azure-sdk/releases/latest/all/python.html))

## Need Help?

*   **Documentation:** Visit the official [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issue Tracking:** Report issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community Support:** Search for answers or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection

The software collects usage data, as described in the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default.  To disable it, create a custom `NoUserAgentPolicy` and pass it during client construction (see example in original README).

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for contribution details. This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).