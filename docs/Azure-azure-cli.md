# Azure CLI: Manage Your Azure Resources from the Command Line

**Empower your cloud management with the Microsoft Azure CLI, a cross-platform command-line interface for interacting with Azure.** [Learn more at the original repo](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:** Works seamlessly across Windows, macOS, and Linux.
*   **Comprehensive Azure Support:** Manage a wide array of Azure services and resources.
*   **Intuitive Command Structure:**  Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:**  Streamlines your workflow with tab completion for commands and parameters.
*   **Flexible Output Formatting:** Supports JSON, table, and TSV output formats, and customizable output with JMESPath queries.
*   **Scripting Friendly:** Provides exit codes for automation and scripting purposes.
*   **VS Code Integration:** Enhance your workflow with the Azure CLI Tools extension for Visual Studio Code, including IntelliSense, snippets, and in-editor command execution.
*   **Edge Builds:** Stay up-to-date with the latest features using edge builds directly from the `dev` branch.

## Getting Started

*   **Installation:**  Follow the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for your operating system. Troubleshoot common issues with the [install troubleshooting guide](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Quick Start:** Get up and running quickly with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Help:** Use the `-h` parameter (e.g., `az storage -h`) for detailed help and usage instructions.

## Usage Examples

*   **List Virtual Machines:** `az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"`
*   **Show VM Details:** `az vm show -g WebPropertiesRG -n Bizlogic`

## Additional Resources

*   **Samples and Snippets:** Explore more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and on the [Microsoft Learn documentation](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Extension:** Install the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) for enhanced development experience.

## Developer Information

*   **Telemetry:** Data collection is enabled by default. You can opt-out with `az config set core.collect_telemetry=no`.
*   **Reporting Issues:**  Report bugs and provide feedback through the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.  Use the `az feedback` command from the command line or contact the developer team at azpycli@microsoft.com.
*   **Developer Setup:** For those interested in contributing, see the [Developer setup](#developer-setup) section.
*   **Contribution Guidelines:**  Learn how to contribute and adhere to the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).