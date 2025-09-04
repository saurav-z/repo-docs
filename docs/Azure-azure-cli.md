# Azure CLI: Your Command-Line Interface for Effortless Cloud Management

**Manage your Azure resources with ease using the Azure CLI, a powerful, cross-platform command-line tool.** Explore this comprehensive guide to discover its key features, installation, and how to contribute to the project.  [Learn more and contribute on GitHub!](https://github.com/Azure/azure-cli)

## Key Features

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Intuitive Syntax:**  `az [group] [subgroup] [command] {parameters}` makes commands easy to understand and execute.
*   **Tab Completion:**  Saves time and reduces errors with tab completion for groups, commands, and parameters.
*   **Powerful Querying:** Utilize the `--query` parameter and JMESPath to customize and filter your output.
*   **Scripting Friendly:** Leverages exit codes for integrating Azure CLI commands into your scripts and automation workflows.
*   **Output Formatting:**  Get the data you need in `json`, `table`, or `tsv` formats.
*   **REST API Command:** Directly interact with Azure REST APIs using the `az rest` command.
*   **Visual Studio Code Integration:** Enhance your workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension, offering features like IntelliSense, snippets, and integrated terminal execution.

## Getting Started

*   **Installation:** Follow the detailed [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for your operating system.  Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Quick Start:**  Get up and running with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Help:**  Access help documentation for any command using the `-h` parameter (e.g., `az storage -h`, `az vm create -h`).

## Developer Installation

*   **Docker:** Use a preconfigured Docker image: `docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>` [See Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list)
*   **Edge Builds:**  Get the latest builds from the `dev` branch:
    *   [MSI](https://aka.ms/InstallAzureCliWindowsEdge)
    *   [Homebrew Formula](https://aka.ms/InstallAzureCliHomebrewEdge)
    *   [Ubuntu Bionic Deb](https://aka.ms/InstallAzureCliBionicEdge)
    *   [Ubuntu Focal Deb](https://aka.ms/InstallAzureCliFocalEdge)
    *   [Ubuntu Jammy Deb](https://aka.ms/InstallAzureCliJammyEdge)
    *   [RPM el8](https://aka.ms/InstallAzureCliRpmEl8Edge)
*   **Get builds of arbitrary commit or PR:** [Try new features before release](doc/try_new_features_before_release.md)
*   **Developer Setup:** Learn how to configure your development environment and contribute:
    *   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
    *   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
    *   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Reporting Issues & Feedback

*   **File an Issue:** Report bugs or suggest improvements in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Command-Line Feedback:** Provide feedback directly from the CLI using the `az feedback` command.
*   **Contact the Team:** \[Microsoft internal]  azpycli@microsoft.com

## Data Collection and Telemetry

The Azure CLI collects data to improve performance and user experience. You can opt-out of telemetry by running `az config set core.collect_telemetry=no`. For more details, please consult the Microsoft privacy statement at https://go.microsoft.com/fwlink/?LinkID=824704.

## Contribute

This project welcomes contributions! See [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for details. This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com)