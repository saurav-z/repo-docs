# Azure SDK for Python: Simplify Cloud Development

**The Azure SDK for Python empowers developers to build robust, scalable, and secure applications on Microsoft Azure.** ([Original Repo](https://github.com/Azure/azure-sdk-for-python))

This repository provides the source code for the Azure SDK for Python, offering a comprehensive suite of libraries to interact with Azure services.

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including storage, compute, networking, databases, and more.
*   **Modern Design:** Built with the [Azure SDK Design Guidelines for Python](https://azure.github.io/azure-sdk/python/guidelines/), offering consistency and a modern developer experience.
*   **Client Libraries:** Utilize dedicated libraries for each Azure service, enabling granular control and efficient resource management.
*   **Management Libraries:** Provision and manage Azure resources effectively with a consistent and intuitive interface.
*   **Production-Ready & Preview Packages:** Choose from stable, production-ready libraries and preview releases with the latest features.
*   **Core Functionality:** Benefit from shared core functionalities like retries, logging, authentication, and HTTP pipeline management.
*   **Simplified Authentication:** Leverage the Azure Identity library for secure and streamlined authentication.
*   **Detailed Documentation:** Access comprehensive documentation and code samples to get you started quickly.

## Getting Started

*   **Prerequisites:** Python 3.9 or later. See [Azure SDK for Python version support policy](https://github.com/Azure/azure-sdk-for-python/wiki/Azure-SDKs-Python-version-support-policy) for details.
*   **Service Libraries:**  Find specific library documentation within each service's `README.md` or `README.rst` file located in the `/sdk` directory.

## Available Packages

The SDK organizes packages into the following categories:

*   **Client - New Releases:** (Generally Available (GA) and preview packages)
*   **Client - Previous Versions:** (Stable, production-ready packages)
*   **Management - New Releases:** (Management libraries following the Azure SDK Design Guidelines)
*   **Management - Previous Versions:** (Management libraries for resource provisioning and management)

### Find the Latest Packages

*   **Client Releases:** [Most up to date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/index.html#python)
*   **Management Releases:** [Most up to date list of all of the new packages](https://azure.github.io/azure-sdk/releases/latest/mgmt/python.html)
*   **All Python Packages:** [Complete list of all libraries](https://azure.github.io/azure-sdk/releases/latest/all/python.html)

## Need Help?

*   **Documentation:** [Azure SDK for Python documentation](https://aka.ms/python-docs)
*   **GitHub Issues:** File issues via [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues)
*   **Stack Overflow:** Search and ask questions on Stack Overflow using the `azure` and `python` tags.

## Data Collection

The software may collect information about you and your use of the software and send it to Microsoft.  You may turn off telemetry. For details, see the [Telemetry Guidelines](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is on by default.  To opt out, use a `NoUserAgentPolicy` class, as shown in the example in the original README, and pass it as an argument during client creation.

## Security

*   **Report Security Issues:** Report security issues privately to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.

## Contributing

*   **Contributing Guide:** See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md).
*   **Contributor License Agreement (CLA):** Required for contributions.
*   **Code of Conduct:** This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).