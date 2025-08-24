# Azure CLI: Manage Azure Resources from the Command Line

**Simplify your cloud management and automation with the Azure CLI, a powerful and versatile command-line interface for interacting with Microsoft Azure.**

**[Explore the Azure CLI Repository](https://github.com/Azure/azure-cli)**

The Azure CLI provides a cross-platform experience, enabling you to manage your Azure resources from Windows, macOS, and Linux.

## Key Features

*   **Cross-Platform Compatibility:** Run on Windows, macOS, and Linux.
*   **Resource Management:** Create, manage, and delete Azure resources.
*   **Automation:** Script complex tasks with ease.
*   **Tab Completion:**  Boost productivity with tab completion for commands and parameters.
*   **Querying with JMESPath:** Customize output with powerful query capabilities.
*   **Output Formatting:** Choose from JSON, table, or TSV formats for easy readability.
*   **REST API Command:** Access the REST API directly using the `az rest` command.
*   **VS Code Integration:**  Enhance your workflow with the Azure CLI Tools extension for Visual Studio Code.

## Get Started

*   **Installation:** Follow the detailed installation guide [here](https://learn.microsoft.com/cli/azure/install-azure-cli). Troubleshoot common installation issues [here](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Quick Start:**  Start with the ["get started" guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).
*   **Basic Command Structure:** Use the `az [group] [subgroup] [command] {parameters}` structure.
*   **Help:** Access help with the `-h` parameter (e.g., `az storage -h` or `az vm create -h`).

## Core Functionality & Effective Use

*   **Tab Completion:** Save time with auto-completion for commands, groups, and parameters.
*   **Querying:** Use the `--query` parameter and [JMESPath](http://jmespath.org/) for customized output.
*   **Exit Codes:** Utilize exit codes for scripting (0: success, 1: generic error, 2: parser error, 3: missing resource).

## Common Scenarios and Tips

*   **Output Formatting:** Control output with JSON, table, or TSV formats.
*   **Passing Values:**  Pass values between commands to chain operations.
*   **Asynchronous Operations:**  Manage asynchronous operations efficiently.
*   **Generic Update Arguments:**  Use consistent arguments for updates.
*   **`az resource` Commands:**  Manage resources with generic resource commands.
*   **`az rest` Commands:**  Interact with the Azure REST API.
*   **Work behind a Proxy:**  Configure the CLI to work behind a proxy server.
*   **Concurrent Builds:** Run concurrent builds with the CLI.
*   **[Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively)**

## Further Resources

*   **Samples & Snippets:**  Explore more usage examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and the [overview documentation](https://learn.microsoft.com/cli/azure/overview).
*   **Visual Studio Code Extension:** Utilize the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) for enhanced development.

## Data Collection & Telemetry

The Azure CLI collects usage data to improve the service. You can opt-out by running `az config set core.collect_telemetry=no`. View the Microsoft privacy statement [here](https://go.microsoft.com/fwlink/?LinkID=824704).

## Reporting Issues & Feedback

*   **Report Bugs:** File issues in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:** Use the `az feedback` command.

## Developer Installation

### Docker
Use preconfigured CLI images. For available versions, see the [Docker tags](https://mcr.microsoft.com/v2/azure-cli/tags/list).

```bash
$ docker run -u $(id -u):$(id -g) -v ${HOME}:/home/az -e HOME=/home/az --rm -it mcr.microsoft.com/azure-cli:<version>
```

### Edge Builds

*   Get the latest builds from the `dev` branch via edge builds.
*   Available builds are provided via the links below:

|      Package      | Link                                       |
|:-----------------:|:-------------------------------------------|
|        MSI        | https://aka.ms/InstallAzureCliWindowsEdge  |
| Homebrew Formula  | https://aka.ms/InstallAzureCliHomebrewEdge |
| Ubuntu Bionic Deb | https://aka.ms/InstallAzureCliBionicEdge   |
| Ubuntu Focal Deb  | https://aka.ms/InstallAzureCliFocalEdge    |
| Ubuntu Jammy Deb  | https://aka.ms/InstallAzureCliJammyEdge    |
|      RPM el8      | https://aka.ms/InstallAzureCliRpmEl8Edge   |

*   **Install with Homebrew:**

    ```bash
    curl --location --silent --output azure-cli.rb https://aka.ms/InstallAzureCliHomebrewEdge
    brew install --build-from-source azure-cli.rb
    ```

*   **Install with Ubuntu Jammy:**

    ```bash
    curl --location --silent --output azure-cli_jammy.deb https://aka.ms/InstallAzureCliJammyEdge && dpkg -i azure-cli_jammy.deb
    ```

*   **Install with RHEL 8 or CentOS Stream 8:**

    ```bash
    dnf install -y $(curl --location --silent --output /dev/null --write-out %{url_effective} https://aka.ms/InstallAzureCliRpmEl8Edge)
    ```

*   **Install with pip3 in a virtual environment:**

    ```bash
    $ python3 -m venv env
    $ . env/bin/activate
    $ pip3 install --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --upgrade-strategy=eager
    ```

*   **To upgrade the edge build:**

    ```bash
    $ pip3 install --upgrade --pre azure-cli --extra-index-url https://azurecliprod.blob.core.windows.net/edge --no-cache-dir --upgrade-strategy=eager
    ```

### Get builds of arbitrary commit or PR

See [Try new features before release](doc/try_new_features_before_release.md)

## Developer Setup

*   **Configuring Your Machine:**  [Configuring Your Machine](https://github.com/Azure/azure-cli/blob/dev/doc/configuring_your_machine.md)
*   **Authoring Command Modules:** [Authoring Command Modules](https://github.com/Azure/azure-cli/tree/dev/doc/authoring_command_modules)
*   **Code Generation:** [Code Generation](https://github.com/Azure/aaz-dev-tools)

## Contribute Code

This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
See the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.
Follow the instructions provided in [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) to contribute.