# Azure CLI: Manage Your Azure Resources from the Command Line

**The Azure CLI is a powerful cross-platform command-line tool for managing and interacting with your Azure resources.** [Learn more and contribute on GitHub](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform:** Compatible with Windows, macOS, and Linux.
*   **Comprehensive:** Supports a vast array of Azure services.
*   **Interactive & Scriptable:**  Execute commands directly or integrate them into scripts for automation.
*   **Tab Completion:** Save time with command, group, and parameter autocompletion.
*   **Flexible Output:** Customize output with JMESPath queries, and choose from JSON, table, or TSV formats.
*   **Azure Cloud Shell Integration:**  Test and use the CLI directly within your browser.
*   **VS Code Extension:** Enhance your development workflow with IntelliSense, snippets, and integrated terminal support.

## Installation

Easily install the Azure CLI on your preferred operating system.  Consult the detailed [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for step-by-step instructions. Resolve common install issues with the [troubleshooting guide](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Getting Started

Use the following command structure to begin:

```bash
$ az [group] [subgroup] [command] {parameters}
```

Get help and explore available commands using the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

For comprehensive guidance, check out the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

## Highlights

The Azure CLI offers several features to boost your productivity:

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

*   **Tab Completion:**  Quickly find commands and parameters.
*   **Query:** Customize output using `--query` and [JMESPath](http://jmespath.org/).
*   **Exit Codes:** Understand command execution outcomes for scripting with well-defined exit codes (0=success, 1=generic error, 2=parser error, 3=missing ARM resource).

## Common Use Cases

Explore these helpful scenarios:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

For more advanced samples, browse the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) or explore [learn.microsoft.com/cli/azure/overview](https://learn.microsoft.com/cli/azure/overview).

## Visual Studio Code Integration

Use the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code for improved development:

*   IntelliSense for commands and arguments.
*   Command snippets for streamlined coding.
*   Run commands in the integrated terminal.
*   View output side-by-side in an editor.
*   Documentation on hover.
*   Display subscription details in the status bar.

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection & Telemetry

The software may collect usage data to improve services. You can disable telemetry by running `az config set core.collect_telemetry=no`.  Refer to Microsoft's privacy statement for details: https://go.microsoft.com/fwlink/?LinkID=824704.

## Reporting Issues & Providing Feedback

Report bugs and provide feedback:

*   File issues in the [GitHub repo](https://github.com/Azure/azure-cli/issues).
*   Use the `az feedback` command from the command line.
*   \[Microsoft internal] Contact the developer team at azpycli@microsoft.com.

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest builds from the `dev` branch. See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for available versions.

| Package          | Link                                      |
| ---------------- | ----------------------------------------- |
| MSI              | https://aka.ms/InstallAzureCliWindowsEdge |
| Homebrew Formula | https://aka.ms/InstallAzureCliHomebrewEdge|
| Ubuntu Bionic Deb| https://aka.ms/InstallAzureCliBionicEdge  |
| Ubuntu Focal Deb | https://aka.ms/InstallAzureCliFocalEdge   |
| Ubuntu Jammy Deb | https://aka.ms/InstallAzureCliJammyEdge   |
| RPM el8          | https://aka.ms/InstallAzureCliRpmEl8Edge   |

Install Homebrew Edge:
```bash
curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
brew install --build-from-source azure-cli.rb
```

Install Ubuntu Jammy Edge:
```bash
curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb
```

Install RPM Edge:
```bash
dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)
```

###  Install with pip3
```bash
$ python3 -m venv env
$ . env/bin/activate
$ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
```

Upgrade Edge:
```bash
$ pip3 install --upgrade --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --no-cache-dir --upgrade-strategy=eager
```

### Get builds of arbitrary commit or PR
See [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

Set up your development environment and contribute to the CLI:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contributing

Contribute to the Azure CLI!  This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).  Learn more in the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com). See [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for contribution instructions.