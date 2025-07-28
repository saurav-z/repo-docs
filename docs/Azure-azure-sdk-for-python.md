# Azure SDK for Python: Simplify Cloud Development in Python

**The Azure SDK for Python provides a comprehensive set of libraries for interacting with Azure services, enabling Python developers to build robust and scalable cloud applications.** [Visit the original repository](https://github.com/Azure/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, databases, networking, and more.
*   **Well-Documented:** Extensive documentation and code samples to get you started quickly.
*   **Consistent API Design:** Libraries follow the Azure SDK Design Guidelines for a familiar and easy-to-use development experience.
*   **Authentication Support:** Seamlessly integrate with Azure Active Directory and other authentication methods.
*   **Management and Client Libraries:** Utilize libraries for both managing Azure resources and building applications that interact with them.
*   **Active Development:** Benefit from ongoing updates, new features, and improvements.

## Getting Started

### Prerequisites

*   Python 3.9 or later

To get started with a specific library, see the `README.md` (or `README.rst`) file located in the library's project folder within the `/sdk` directory.

## Packages Available

Each service has libraries available in these categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest **GA** (Generally Available) and **Preview** libraries for interacting with Azure services. They share core functionalities like retries, logging, and authentication.

*   [Most up-to-date package list](https://azure.github.io/azure-sdk/releases/latest/index.html#python)

> **Important:** For production use, choose stable, non-preview libraries.

### Client: Previous Versions

Stable, production-ready packages for interacting with Azure. They may offer wider service coverage, but may not implement the latest guidelines or features.

### Management: New Releases

Management libraries follow the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/) for managing Azure resources.

*   Documentation and samples: [here](https://aka.ms/azsdk/python/mgmt)
*   Migration guide: [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide)
*   [Most up-to-date package list](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)

> **Important:** If you're experiencing authentication issues after upgrading management packages, consult the migration guide.

### Management: Previous Versions

Libraries for provisioning and managing Azure resources.

*   [Complete list](https://azure.github.io/azure-sdk/releases/latest/all/python.html)
*   Identified by namespaces starting with `azure-mgmt-`, e.g. `azure-mgmt-compute`.

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   Stack Overflow: Use tags `azure` and `python`.

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft. For more information on the data collected by the Azure SDK, please visit the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy) page.

### Telemetry Configuration

Telemetry is enabled by default. To opt-out, use the `NoUserAgentPolicy` class, which disables telemetry.
See the example in the original README for implementation.

## Reporting Security Issues

Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md). This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).