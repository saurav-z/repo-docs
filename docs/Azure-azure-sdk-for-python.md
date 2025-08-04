# Azure SDK for Python: Build Powerful Python Applications for the Cloud

**Get started with the Azure SDK for Python to seamlessly integrate your Python applications with Microsoft Azure services and unlock the power of the cloud.**  Learn more about this powerful SDK on the [original repository](https://github.com/Azure/azure-sdk-for-python).

## Key Features

*   **Comprehensive Service Coverage:** Access a wide range of Azure services, including compute, storage, networking, databases, and more.
*   **Client and Management Libraries:** Utilize both client libraries for interacting with existing resources and management libraries for provisioning and managing Azure resources.
*   **Modern Design Guidelines:** Benefit from libraries built following the Azure SDK Design Guidelines for Python, ensuring consistency and ease of use.
*   **Cross-Platform Compatibility:**  Supports Python 3.9 and later, enabling you to develop and deploy applications across various platforms.
*   **Built-in Features:** Leverage shared core functionalities like retries, logging, transport and authentication protocols, and more.
*   **Extensive Documentation and Samples:** Find comprehensive documentation, code samples, and migration guides to help you get started quickly.

## Getting Started

Each Azure service has its own dedicated Python libraries, offering a modular approach.  Explore the `/sdk` directory to discover service-specific libraries and their corresponding `README.md` or `README.rst` files for detailed instructions.

## Key Areas

### Client Libraries

*   **Client: New Releases:** GA (Generally Available) and preview libraries for consuming existing resources. Learn more about these new releases on the [Azure SDK releases page](https://azure.github.io/azure-sdk/releases/latest/index.html#python).
*   **Client: Previous Versions:** Stable, production-ready libraries providing similar functionalities as preview versions with wider service coverage.

### Management Libraries

*   **Management: New Releases:** Provision and manage Azure resources with the latest management libraries following Azure SDK Design Guidelines. Find documentation and samples [here](https://aka.ms/azsdk/python/mgmt).
*   **Management: Previous Versions:**  Complete list of management libraries [here](https://azure.github.io/azure-sdk/releases/latest/all/python.html) to manage Azure resources, identified by `azure-mgmt-` namespaces.

## Need Help?

*   **Documentation:** Explore the comprehensive [Azure SDK for Python documentation](https://aka.ms/python-docs).
*   **Issue Tracking:** Report issues on [GitHub Issues](https://github.com/Azure/azure-sdk-for-python/issues).
*   **Community Support:**  Search for or ask questions on StackOverflow using the `azure` and `python` tags.

## Data Collection & Telemetry

The Azure SDK for Python collects data to improve services.  Learn more about data collection and Microsoft's privacy statement [here](https://go.microsoft.com/fwlink/?LinkID=824704), and about telemetry specifically for the Azure SDK [here](https://azure.github.io/azure-sdk/general_azurecore.html#telemetry-policy).

### Telemetry Configuration

Telemetry is enabled by default. You can disable telemetry by creating a `NoUserAgentPolicy` and passing it during client creation.

## Security

*   Report security issues to the Microsoft Security Response Center (MSRC) at <secure@microsoft.com>.  See the [Security TechCenter](https://www.microsoft.com/msrc/faqs-report-an-issue) for further details.

## Contributing

See the [contributing guide](https://github.com/Azure/azure-sdk-for-python/blob/main/CONTRIBUTING.md) for details on how to contribute.  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).