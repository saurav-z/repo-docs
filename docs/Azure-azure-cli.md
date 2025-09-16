# Azure CLI: Command-Line Interface for Azure Cloud Management

**Manage and automate your Azure cloud resources with ease using the powerful and versatile Azure CLI.** ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform Compatibility:** Use the Azure CLI on Windows, macOS, and Linux.
*   **Comprehensive Resource Management:** Control and configure a wide array of Azure services.
*   **Interactive and Scriptable:** Execute commands directly or integrate them into your scripts and automation pipelines.
*   **Powerful Querying:** Utilize JMESPath queries for precise output formatting and data extraction.
*   **Tab Completion:** Benefit from tab completion for commands and parameters, streamlining your workflow.
*   **Customizable Output:** Choose from various output formats, including JSON, table, and tsv, to suit your needs.
*   **Exit Codes for Scripting:** Leverage consistent exit codes for reliable scripting and automation.
*   **VS Code Integration:** Enhance your development experience with the Azure CLI Tools extension for Visual Studio Code, offering IntelliSense, snippets, and more.

## Installation

Refer to the comprehensive [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions. Troubleshoot common installation issues using the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) guide.

## Usage

Use the `az` command followed by the resource group, command, and parameters to manage your Azure resources:

```bash
$ az [ group ] [ subgroup ] [ command ] {parameters}
```

For example:
```bash
$ az storage -h
$ az vm create -h
```

## Get Started

Explore the Azure CLI with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

## Highlights

*   **Tab completion**
*   **Query output**
*   **Exit Codes**

## Common Scenarios and Effective Usage

Learn best practices and examples for using the Azure CLI efficiently:

*   [Output formatting](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Samples and Snippets

Explore additional usage examples in our [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and on the [Microsoft Learn page](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

Enhance your development workflow with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code.

## Data Collection

The software collects data to help provide services and improve products and services. You may turn off telemetry, as described in the repository. [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704)

### Telemetry Configuration

Telemetry collection is on by default. To opt out, please run `az config set core.collect_telemetry=no` to turn it off.

## Reporting Issues and Feedback

Report bugs and provide feedback through the [Issues](https://github.com/Azure/azure-cli/issues) section.
Provide feedback from the command line with the `az feedback` command.

## Developer Installation

### Docker
See our [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for available versions.
```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds
Edge builds are available for the latest builds from the `dev` branch.

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

### Get builds of arbitrary commit or PR

See:
[Try new features before release](doc/try_new_features_before_release.md)

## Developer setup

For contributions see:

[Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)

[Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)

[Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

Contribute code following the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).