# Azure CLI: Command-Line Interface for Azure (SEO-Optimized)

**Effortlessly manage and control your Azure cloud resources with the powerful and versatile Azure CLI.**  [Explore the Azure CLI Repository on GitHub](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:**  Works seamlessly across Windows, macOS, and Linux.
*   **Simplified Cloud Management:** Automate and streamline your Azure tasks with easy-to-use commands.
*   **Interactive Cloud Shell Support:**  Get started quickly with in-browser access to the CLI via the Azure Cloud Shell.
*   **Tab Completion:**  Enhance productivity with intelligent tab completion for commands, groups, and parameters.
*   **Flexible Output Formatting:**  Customize your output with JSON, table, and TSV formats.
*   **Powerful Querying:**  Use JMESPath queries to precisely filter and format your results.
*   **Scripting Friendly:**  Leverage exit codes for seamless integration with scripts and automation workflows.
*   **REST API Integration:** Execute REST API calls directly using the `az rest` command.
*   **VS Code Integration:** The [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension enhances your productivity in Visual Studio Code with features like IntelliSense, snippets, and documentation.

## Get Started

### Installation

Install the Azure CLI quickly and easily using the detailed instructions in the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

Troubleshooting resources are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

### Basic Usage

The general command structure is:

```bash
$ az [group] [subgroup] [command] {parameters}
```

**Example:**

```bash
$ az vm create -g MyResourceGroup -n MyVM --image UbuntuLTS
```

For in-depth guidance, refer to the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).  Use the `-h` parameter for help on specific commands:

```bash
$ az storage -h
$ az vm create -h
```

## Advanced Features

*   **Tab Completion:**  Improve efficiency with tab completion for groups, commands, and parameters.
*   **Querying:**  Use the `--query` parameter and [JMESPath](http://jmespath.org/) to customize the output.
*   **Exit Codes:**  Understand exit codes for scripting and automation.

## Common Scenarios and Effective Usage

Enhance your CLI usage with these tips:

*   [Output formatting (JSON, table, or TSV)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Developer Installation & Contribution

### Docker

The Azure CLI is available as a preconfigured Docker image:

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Stay up-to-date with the latest developments using edge builds.

*   **MSI:** [https://aka.ms/InstallAzureCliWindowsEdge](https://aka.ms/InstallAzureCliWindowsEdge)
*   **Homebrew:** [https://aka.ms/InstallAzureCliHomebrewEdge](https://aka.ms/InstallAzureCliHomebrewEdge)
*   **Ubuntu Bionic Deb:** [https://aka.ms/InstallAzureCliBionicEdge](https://aka.ms/InstallAzureCliBionicEdge)
*   **Ubuntu Focal Deb:** [https://aka.ms/InstallAzureCliFocalEdge](https://aka.ms/InstallAzureCliFocalEdge)
*   **Ubuntu Jammy Deb:** [https://aka.ms/InstallAzureCliJammyEdge](https://aka.ms/InstallAzureCliJammyEdge)
*   **RPM el8:** [https://aka.ms/InstallAzureCliRpmEl8Edge](https://aka.ms/InstallAzureCliRpmEl8Edge)

### Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)

## Contribute

Contribute to the Azure CLI project by following the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).  See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) for questions.

## Reporting Issues and Providing Feedback

Report bugs and provide feedback through the [Issues](https://github.com/Azure/azure-cli/issues) section of the GitHub repository or use the `az feedback` command.