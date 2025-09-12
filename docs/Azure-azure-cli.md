# Azure CLI: Command-Line Interface for Azure Cloud

**Manage your Azure resources efficiently with the Azure CLI, a cross-platform command-line tool for interacting with Microsoft Azure.**

[See the original repo here](https://github.com/Azure/azure-cli)

## Key Features

*   **Cross-Platform:** Works seamlessly on Windows, macOS, and Linux.
*   **Resource Management:** Create, manage, and delete Azure resources directly from the command line.
*   **Tab Completion:** Enhance productivity with tab completion for commands, groups, and parameters.
*   **Querying:** Use JMESPath queries with the `--query` parameter to customize output and filter data.
*   **Output Formatting:** Customize your output with options like JSON, table, or TSV.
*   **Scripting Support:** Leverage exit codes for scripting and automation.
*   **REST API Integration:** Access the Azure REST API with the `az rest` command.
*   **Visual Studio Code Integration:** Integrate with the Azure CLI Tools extension for enhanced development experience.

## Installation

Easily install the Azure CLI on your preferred operating system. Detailed instructions are available in the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli).

Troubleshooting common installation issues is available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

### Getting Started

Get up and running quickly with the Azure CLI by following the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
Explore commands and their parameters using the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

## Core Usage

The general structure of an Azure CLI command is:

```bash
$ az [group] [subgroup] [command] {parameters}
```

## Highlights & Examples

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

### Tab Completion Example:

```bash
$ az vm show -g [tab][tab]
AccountingGroup   RGOne  WebPropertiesRG

$ az vm show -g WebPropertiesRG -n [tab][tab]
StoreVM  Bizlogic

$ az vm show -g WebPropertiesRG -n Bizlogic
```

### Querying Example:

```bash
$ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
Name                    Os
----------------------  -------
storevm                 Linux
bizlogic                Linux
demo32111vm             Windows
dcos-master-39DB807E-0  Linux
```

### Exit Codes for Scripting

| Exit Code | Scenario                                          |
| --------- | ------------------------------------------------- |
| 0         | Command ran successfully.                        |
| 1         | Generic error (bad status code, validation fail). |
| 2         | Parser error; check input.                       |
| 3         | Missing ARM resource.                            |

## Advanced Usage & Tips

Explore these resources to use the Azure CLI effectively:

*   [Output Formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## Samples and Snippets

Find more usage examples in our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and the comprehensive documentation: [https://learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for VS Code

Enhance your development workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code. This extension provides:

*   IntelliSense for commands and arguments.
*   Command snippets.
*   In-terminal command execution.
*   Output display in side-by-side editors.
*   Documentation on hover.
*   Subscription and default display in the status bar.

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Developer Installation

### Docker

Preconfigured Docker images are available to use the Azure CLI.
See our [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for available versions.

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds from the `dev` branch.

| Package           | Link                                        |
| :---------------- | :------------------------------------------ |
| MSI               | https://aka.ms/InstallAzureCliWindowsEdge   |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge  |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge    |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge     |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge     |
| RPM el8           | https://aka.ms/InstallAzureCliRpmEl8Edge    |

#### Homebrew Example

```bash
curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
brew install --build-from-source azure-cli.rb
```

#### Ubuntu Jammy Example

```bash
curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb
```

#### RPM Example

```bash
dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)
```

### Build of Arbitrary Commits and PRs

[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

If you'd like to contribute, check out the following resources:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute

This project welcomes contributions and follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Find more information in the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.  To contribute, follow the instructions in the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate).

## Reporting Issues and Feedback

Report any bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section. Provide feedback from the command line with `az feedback`.

\[Microsoft internal]  Contact the dev team via azpycli@microsoft.com.