# Azure CLI: Command-Line Interface for Managing Azure Resources

**Effortlessly manage and automate your Azure cloud resources with the Azure CLI, a powerful and versatile cross-platform command-line interface.** ([Back to the original repo](https://github.com/Azure/azure-cli))

## Key Features:

*   **Cross-Platform Compatibility:** Available on Windows, macOS, and Linux.
*   **Comprehensive Azure Coverage:** Supports a wide range of Azure services.
*   **Intuitive Command Structure:** Uses a clear and consistent `az [group] [subgroup] [command] {parameters}` syntax.
*   **Tab Completion:**  Speeds up command entry with tab completion for groups, commands, and parameters.
*   **Flexible Output Formatting:**  Customize output with JSON, table, or TSV formats, and use `--query` with JMESPath for powerful data extraction.
*   **Scripting Friendly:**  Provides exit codes for easy integration into scripts and automation workflows.
*   **Integrated with VS Code:** Enhance your experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension.
*   **Regular Updates:** Stay up-to-date with the latest Azure features and enhancements.

## Getting Started

### Installation

Refer to the detailed [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for instructions. Troubleshooting tips are available in the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) document.

### Basic Usage

Use the following structure for commands:

```bash
$ az [group] [subgroup] [command] {parameters}
```

Explore commands using the `-h` parameter for help:

```bash
$ az storage -h
$ az vm create -h
```

### Useful Scenarios

Explore effective usage with these guides:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)

### Examples

*   **Tab Completion:**

    ```bash
    # looking up resource group and name
    $ az vm show -g [tab][tab]
    AccountingGroup   RGOne  WebPropertiesRG

    $ az vm show -g WebPropertiesRG -n [tab][tab]
    StoreVM  Bizlogic

    $ az vm show -g WebPropertiesRG -n Bizlogic
    ```

*   **Querying for specific data:**

    ```bash
    $ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
    Name                    Os
    ----------------------  -------
    storevm                 Linux
    bizlogic                Linux
    demo32111vm             Windows
    dcos-master-39DB807E-0  Linux
    ```

## Further Resources

*   **Samples:**  Explore more examples and snippets at the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).
*   **VS Code Integration:**  Use the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for enhanced development with IntelliSense, snippets, and integrated terminal support.

## Developer Information

*   **Docker:** Preconfigured images are available - see [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).
*   **Edge Builds:** Access the latest builds from the `dev` branch:
    *   [MSI](https://aka.ms/InstallAzureCliWindowsEdge)
    *   [Homebrew Formula](https://aka.ms/InstallAzureCliHomebrewEdge)
    *   [Ubuntu Bionic Deb](https://aka.ms/InstallAzureCliBionicEdge)
    *   [Ubuntu Focal Deb](https://aka.ms/InstallAzureCliFocalEdge)
    *   [Ubuntu Jammy Deb](https://aka.ms/InstallAzureCliJammyEdge)
    *   [RPM el8](https://aka.ms/InstallAzureCliRpmEl8Edge)
*   **Developer Setup:** See [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md) and [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules) for setting up a development environment.
*   **Contribute:**  Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and the [Code of Conduct](https://opensource.microsoft.com/codeofconduct/) to contribute code.

## Support and Feedback

*   **Issues:** Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Feedback:** Provide command-line feedback with the `az feedback` command.