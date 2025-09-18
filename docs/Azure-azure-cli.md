# Azure CLI: Command-Line Interface for Azure Management

**Simplify your Azure cloud management with the powerful and versatile Azure CLI, a cross-platform command-line tool.**  ([Back to Original Repo](https://github.com/Azure/azure-cli))

## Key Features

*   **Cross-Platform:** Available on Windows, macOS, and Linux.
*   **Comprehensive:** Manage a wide array of Azure services.
*   **Easy to Use:** Intuitive syntax with tab completion for efficient command execution.
*   **Powerful Querying:** Utilize JMESPath queries for customized output.
*   **Scripting-Friendly:** Output exit codes for seamless integration into scripts.
*   **Extensible:** Integrates with Visual Studio Code for enhanced development and command execution with the Azure CLI Tools extension.
*   **Up-to-date:** Access and test the newest features by using edge builds.

## Installation

Refer to the [Azure CLI install guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed installation instructions.
Troubleshooting steps are available in the [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md) section.

## Usage

```bash
$ az [group] [subgroup] [command] {parameters}
```

### Get Started

For comprehensive instructions, consult the ["Get Started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

To view usage and help information for a command, use the `-h` parameter:

```bash
$ az storage -h
$ az vm create -h
```

### Highlights

![Azure CLI Highlight Reel](doc/assets/AzBlogAnimation4.gif)

The examples use the `--output table` format, which you can customize with the `az configure` command.

#### Tab Completion

Supports tab completion for groups, commands, and some parameters.

```bash
# Resource group and name lookup
$ az vm show -g [tab][tab] 
AccountingGroup   RGOne  WebPropertiesRG

$ az vm show -g WebPropertiesRG -n [tab][tab]
StoreVM  Bizlogic

$ az vm show -g WebPropertiesRG -n Bizlogic
```

#### Query

Use the `--query` parameter and [JMESPath](http://jmespath.org/) to format your output.

```bash
$ az vm list --query "[?provisioningState=='Succeeded'].{ name: name, os: storageProfile.osDisk.osType }"
Name                    Os
----------------------  -------
storevm                 Linux
bizlogic                Linux
demo32111vm             Windows
dcos-master-39DB807E-0  Linux
```

#### Exit Codes

Important exit codes for scripting purposes:

| Exit Code | Scenario                                                        |
| :-------- | :-------------------------------------------------------------- |
| 0         | Command ran successfully.                                      |
| 1         | Generic error (server error, CLI validation failure, etc.).   |
| 2         | Parser error (input issue).                                    |
| 3         | Missing ARM resource (used for existence checks from `show`). |

### Common Scenarios and Effective Usage

Consult ["Tips for using Azure CLI effectively"](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively) for common scenarios:

*   Output formatting (json, table, or tsv)
*   Pass values between commands
*   Async operations
*   Generic update arguments
*   Generic resource commands (`az resource`)
*   REST API command (`az rest`)
*   Quoting issues
*   Working behind a proxy
*   Concurrent builds

### More Examples and Snippets

*   [GitHub samples repo](http://github.com/Azure/azure-cli-samples)
*   [Azure CLI overview](https://learn.microsoft.com/cli/azure/overview)

### Azure CLI Tools for Visual Studio Code

Use the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) extension to enhance your workflow:

*   IntelliSense for commands and arguments
*   Command snippets
*   Run commands in the integrated terminal
*   Side-by-side output display
*   Documentation on hover
*   Subscription and defaults display in status bar

![Azure CLI Tools in Action](https://github.com/microsoft/vscode-azurecli/blob/main/images/in_action.gif?raw=true)

## Data Collection

The software collects data to improve services; you can disable telemetry with `az config set core.collect_telemetry=no`.  See the [privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704) for details.

## Reporting Issues and Feedback

*   File issues:  [Issues](https://github.com/Azure/azure-cli/issues)
*   Feedback: `az feedback` command
*   Internal Contact:  azpycli@microsoft.com (for Microsoft internal use)

## Developer Installation

### Docker

Preconfigured Azure CLI Docker image.

*   See [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list) for versions.
```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

Get the latest build from the `dev` branch.

*   MSI: https://aka.ms/InstallAzureCliWindowsEdge
*   Homebrew: https://aka.ms/InstallAzureCliHomebrewEdge
*   Ubuntu Bionic Deb: https://aka.ms/InstallAzureCliBionicEdge
*   Ubuntu Focal Deb: https://aka.ms/InstallAzureCliFocalEdge
*   Ubuntu Jammy Deb: https://aka.ms/InstallAzureCliJammyEdge
*   RPM el8: https://aka.ms/InstallAzureCliRpmEl8Edge

Install Homebrew Edge build:
```bash
curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
brew install --build-from-source azure-cli.rb
```

Install Ubuntu Jammy Edge build:

```bash
curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb
```

Install RPM EL8 Edge build:

```bash
dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)
```

Install Edge builds using pip3:

```bash
$ python3 -m venv env
$ . env/bin/activate
$ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
```

Upgrade current edge build:

```bash
$ pip3 install --upgrade --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --no-cache-dir --upgrade-strategy=eager
```

### Get builds of arbitrary commit or PR

*   See: [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

*   [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/)
*   [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
*   Contact: opencode@microsoft.com
*   [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate)