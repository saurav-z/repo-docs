# Azure SDK for Python: Build Powerful Python Applications with Microsoft Azure

**Enhance your Python applications with seamless integration to Microsoft Azure services using the Azure SDK for Python.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

This repository provides the source code and resources for the official Azure SDK for Python. This SDK offers a comprehensive suite of libraries for interacting with various Azure services, enabling developers to build robust, scalable, and cloud-native applications.

**Key Features:**

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Modern Design:** Leverage the latest Azure SDK design guidelines for consistent and intuitive APIs.
*   **Cross-Platform Compatibility:** Build and deploy applications on any platform that supports Python 3.9 or later.
*   **Robust Authentication:** Benefit from secure and streamlined authentication methods, including Azure Active Directory and managed identities.
*   **Simplified Development:** Utilize well-documented libraries with clear examples to accelerate development.
*   **Improved Features:** The new SDKs share several core functionalities such as: retries, logging, transport protocols, authentication protocols, etc.

## Getting Started

To begin, select the specific service library you need from the `/sdk` directory. Each library includes its own `README.md` (or `README.rst`) file with detailed instructions and examples.

## Available Packages

The Azure SDK for Python organizes its libraries into the following categories:

*   [Client - New Releases](#client-new-releases)
*   [Client - Previous Versions](#client-previous-versions)
*   [Management - New Releases](#management-new-releases)
*   [Management - Previous Versions](#management-previous-versions)

### Client: New Releases

These are the latest GA (Generally Available) and preview libraries for interacting with Azure resources. These libraries allow you to use and consume existing resources and interact with them, for example: upload a blob.

*   Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python).

### Client: Previous Versions

These libraries offer stable versions of packages for use with Azure.

### Management: New Releases

Management libraries provide the tools to provision and manage Azure resources, following the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/). These new libraries provide a number of core capabilities that are shared amongst all Azure SDKs, including the intuitive Azure Identity library, an HTTP Pipeline with custom policies, error-handling, distributed tracing, and much more.

*   Documentation and code samples [here](https://aka.ms/azsdk/python/mgmt).
*   Migration guide [here](https://github.com/Azure/azure-sdk-for-python/blob/main/doc/sphinx/mgmt_quickstart.rst#migration-guide).
*   Find the [most up-to-date list of packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html).

### Management: Previous Versions

These libraries provide access to provision and manage Azure resources.

*   Complete list [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html).

## Need Help?

*   [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   StackOverflow using `azure` and `python` tags.

## Data Collection & Telemetry

The SDK collects data to improve services.

*   More information: [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy)
*   Telemetry can be disabled by implementing a `NoUserAgentPolicy` as a subclass of `UserAgentPolicy`, with an empty `on_request` method, and passing it during client creation.

## Reporting Security Issues

Report security vulnerabilities to the Microsoft Security Response Center (MSRC) <secure@microsoft.com>.

## Contributing

Contribute to the Azure SDK for Python by following the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).