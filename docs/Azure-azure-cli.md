# Azure CLI: Manage Your Azure Resources from the Command Line

**Simplify your cloud management with the Azure CLI, a powerful and versatile command-line interface for interacting with Microsoft Azure.**  [View the original repository](https://github.com/Azure/azure-cli).

## Key Features

*   **Cross-Platform Compatibility:**  Works seamlessly on Windows, macOS, and Linux.
*   **Intuitive Command Structure:** Uses a clear `az [group] [subgroup] [command] {parameters}` syntax.
*   **Interactive Shell:** Easily execute commands and get immediate feedback.
*   **Tab Completion:**  Streamline your workflow with intelligent tab completion for commands and parameters.
*   **Flexible Output Options:**  Format results in JSON, table, or TSV formats, or customize with JMESPath queries.
*   **Scripting Support:** Leverage exit codes for robust automation in your scripts.
*   **Extensible Functionality:**  Utilize the `az rest` command for direct REST API interactions.
*   **VS Code Integration:** Enhance your coding experience with the Azure CLI Tools extension for Visual Studio Code, offering features like IntelliSense, snippets, and in-editor command execution.
*   **Azure Cloud Shell Ready:** Test drive the Azure CLI directly in your browser using the Azure Cloud Shell.

## Installation

Follow the detailed installation guide for your platform: [Install Guide](https://learn.microsoft.com/cli/azure/install-azure-cli).
Troubleshooting tips can be found here: [Install Troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).

## Usage

General command structure:
```bash
$ az [group] [subgroup] [command] {parameters}
```

### Get Started

Explore comprehensive instructions with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

### Get Help

Use the `-h` parameter to access help for a specific group, command, or subgroup.  For example:
```bash
$ az storage -h
$ az vm create -h
```

### Highlights & Examples

Learn the most out of the Azure CLI with the following features and concepts:

<img src="doc/assets/AzBlogAnimation4.gif" alt="Azure CLI Highlight Reel">

Show output in a table format with `--output table` (you can change default using `az configure`).

#### Tab Completion

Make your commands simpler with tab completion:

```bash
# looking up resource group and name
$ az vm show -g [tab][tab]
AccountingGroup   RGOne  WebPropertiesRG

$ az vm show -g WebPropertiesRG -n [tab][tab]
StoreVM  Bizlogic

$ az vm show -g WebPropertiesRG -n Bizlogic
```

#### Query

Use the `--query` parameter with [JMESPath](http://jmespath.org/) to customize the output:

```bash
$ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
Name                    Os
----------------------  -------
storevm                 Linux
bizlogic                Linux
demo32111vm             Windows
dcos-master-39DB807E-0  Linux
```

#### Exit codes

See the following exit codes for scripting purposes:

| Exit Code | Scenario                                                      |
| --------- | ------------------------------------------------------------- |
| 0         | Command ran successfully.                                     |
| 1         | Generic error; server returned bad status code, CLI validation failed, etc. |
| 2         | Parser error; check input to command line.                   |
| 3         | Missing ARM resource; used for existence check from `show` commands. |

## Common Scenarios

Explore these common scenarios:

*   [Output formatting (json, table, or tsv)](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#output-formatting-json-table-or-tsv)
*   [Pass values from one command to another](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#pass-values-from-one-command-to-another)
*   [Async operations](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#async-operations)
*   [Generic update arguments](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-update-arguments)
*   [Generic resource commands - `az resource`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#generic-resource-commands---az-resource)
*   [REST API command - `az rest`](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#rest-api-command---az-rest)
*   [Quoting issues](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#quoting-issues)
*   [Work behind a proxy](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#work-behind-a-proxy)
*   [Concurrent builds](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively#concurrent-builds)

## More Samples & Snippets

Discover more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and the [Microsoft documentation](https://learn.microsoft.com/cli/azure/overview).

## Azure CLI Tools for Visual Studio Code

The [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension for Visual Studio Code offers:

*   IntelliSense
*   Snippets
*   Run in integrated terminal
*   Show output in a side-by-side editor
*   Documentation on hover
*   Display of current subscription

<img src="https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true" alt="Azure CLI Tools in Action">

## Data Collection

The software collects data about you and your use of the software and sends it to Microsoft. You can disable telemetry:  `az config set core.collect_telemetry=no`.  For details, refer to the [Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues & Feedback

Report bugs in the [Issues](https://github.com/Azure/azure-cli/issues) section.  Provide feedback via the `az feedback` command.

\[Microsoft internal] Contact the developer team at azpycli@microsoft.com.

## Developer Installation

### Docker

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

```bash
# Homebrew
curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
brew install --build-from-source azure-cli.rb

# Ubuntu Jammy
curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb

# RHEL 8 / CentOS Stream 8
dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)

# pip3 in a virtual environment
$ python3 -m venv env
$ . env/bin/activate
$ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
```

### Get builds of arbitrary commit or PR

[Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

See for development:

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

Follow the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) and [Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with questions.