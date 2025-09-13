# Azure CLI: Manage Azure Resources from the Command Line

**The Azure CLI is a cross-platform command-line experience that provides powerful tools to manage your Azure resources.** ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform:** Works on Windows, macOS, and Linux.
*   **Resource Management:** Create, manage, and configure Azure services with ease.
*   **Tab Completion:** Streamline your workflow with command, group, and parameter completion.
*   **Powerful Querying:** Use JMESPath for customized output with the `--query` parameter.
*   **Scripting Friendly:** Standardized exit codes for reliable automation.
*   **Output Formatting:** Supports JSON, table, and TSV formats for versatile data presentation.
*   **REST API Access:**  Interact directly with Azure REST APIs using `az rest`.
*   **Visual Studio Code Integration:**  Use the Azure CLI Tools extension for IntelliSense, snippets, and integrated terminal access.

## Installation

Refer to the [install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions.  Troubleshooting tips are available in the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) document.

## Usage

```bash
$ az [group] [subgroup] [command] {parameters}
```

### Getting Started

For a comprehensive guide, see the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

Get help using the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

### Highlights

The Azure CLI offers many features to help you manage your Azure resources, like tab completion and powerful querying.

#### Tab Completion

```bash
$ az vm show -g [tab][tab]  # Example
```

#### Query

Customize your output with the `--query` parameter and JMESPath:

```bash
$ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
```

#### Exit Codes

| Exit Code | Scenario                                        |
| --------- | ----------------------------------------------- |
| 0         | Command ran successfully.                       |
| 1         | Generic error, server error, CLI validation failed |
| 2         | Parser error, check input.                      |
| 3         | Missing ARM resource.                           |

### Common Scenarios

Learn how to use the Azure CLI effectively: [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

## More Samples and Snippets

*   Explore more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and on [Microsoft Learn](https://learn.microsoft.com/cli/azure/overview).

## Visual Studio Code Integration

The [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension enhances your VS Code experience with features like IntelliSense and integrated terminal execution.

## Data Collection

The software collects information. You can opt-out with: `az config set core.collect_telemetry=no`.  See our [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues and Feedback

Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section.  Provide feedback via the `az feedback` command.

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds:

| Package          | Link                                       |
| :--------------- | :------------------------------------------- |
| MSI              | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
| RPM el8          | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Get builds of arbitrary commit or PR

See: [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

See:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

See the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/) and [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).