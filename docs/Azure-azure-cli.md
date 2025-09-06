# Azure CLI: Manage Your Azure Resources with Ease

**The Azure CLI is a powerful, cross-platform command-line interface (CLI) for managing and interacting with your Azure cloud resources.**  [Learn more on GitHub](https://github.com/Azure/azure-cli).

[![Python](https://img.shields.io/pypi/pyversions/azure-cli.svg?maxAge=2592000)](https://pypi.python.org/pypi/azure-cli)
[![Build Status](https://dev.azure.com/azure-sdk/public/_apis/build/status/cli/Azure.azure-cli?branchName=dev)](https://dev.azure.com/azure-sdk/public/_build/latest?definitionId=246&branchName=dev)
[![Slack](https://img.shields.io/badge/Slack-azurecli.slack.com-blue.svg)](https://azurecli.slack.com)

**Key Features:**

*   **Cross-Platform:** Available on Windows, macOS, and Linux.
*   **Comprehensive:**  Supports a vast array of Azure services.
*   **Scriptable:**  Automate your Azure tasks with ease.
*   **Tab Completion:**  Saves time and reduces errors with command completion.
*   **Flexible Output:**  Customize output with `--query` and JMESPath for precise data retrieval, supporting formats like JSON, table, or tsv.
*   **IntelliSense Support:**  Boost productivity with the Azure CLI Tools Visual Studio Code extension, offering IntelliSense, snippets, and more.

**Getting Started:**

*   **Installation:**  Follow the [installation guide](https://learn.microsoft.com/cli/azure/install-azure-cli) for detailed instructions.  Troubleshooting tips are available at [install troubleshooting](https://github.com/Azure/azure-cli/blob/dev/doc/install_troubleshooting.md).
*   **Usage:**  Use the following command structure:
    ```bash
    $ az [ group ] [ subgroup ] [ command ] {parameters}
    ```
*   **Help:** Get help with the `-h` parameter (e.g., `az storage -h` or `az vm create -h`).
*   **Get Started Guide:**  Find in-depth instructions on the [get started guide](https://learn.microsoft.com/cli/azure/get-started-with-az-cli2).

**Highlights & Advanced Usage:**

*   **Tab Completion:** Quickly find the groups, commands and parameters with the tab key.
*   **Query:** Use the `--query` parameter with [JMESPath](http://jmespath.org/) to filter and format results.
*   **Exit Codes:** Understand exit codes for scripting: 0 (success), 1 (generic error), 2 (parser error), 3 (missing resource).
*   **Effective Use:** Explore scenarios like output formatting, passing values between commands, async operations, and more using [Tips for using Azure CLI effectively](https://learn.microsoft.com/en-us/cli/azure/use-cli-effectively).

**More Resources:**

*   **Samples:** Explore more examples in the [GitHub samples repo](http://github.com/Azure/azure-cli-samples) and [Microsoft Learn](https://learn.microsoft.com/cli/azure/overview).
*   **Visual Studio Code Extension:** Enhance your experience with the [Azure CLI Tools](https://marketplace.visualstudio.com/items?itemName=ms-vscode.azurecli) for IntelliSense, snippets, and more.

**Developer Information:**

*   **Developer Installation:**
    *   [Docker](#docker)
    *   [Edge Builds](#edge-builds)
    *   [Developer Setup](#developer-setup)
*   **Contribute:**  See the [Microsoft Open Source Guidelines](https://opensource.microsoft.com/collaborate) for contribution guidelines.
*   **Code of Conduct:** This project follows the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/). Contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any questions.

**Feedback and Support:**

*   **Report Issues:**  File issues in the [GitHub Issues](https://github.com/Azure/azure-cli/issues) section.
*   **Provide Feedback:** Use the `az feedback` command from the command line.
*   \[Microsoft internal] Contact the developer team via azpycli@microsoft.com.

**Data Collection and Telemetry:**

*   The Azure CLI collects telemetry data by default. To opt out, run `az config set core.collect_telemetry=no`. More information is available in the help documentation and the [Microsoft privacy statement](https://go.microsoft.com/fwlink/?LinkID=824704).