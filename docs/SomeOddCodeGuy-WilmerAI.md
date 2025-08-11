# WilmerAI: Expertly Routing Your Language Model Inferences

🚀 **Unlock the power of multi-LLM workflows with WilmerAI, an application that orchestrates complex interactions between your front-end and various language model APIs.**

[View the Original Repository](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Prompt Routing:** Direct prompts to custom categories (coding, math, personas, etc.) or directly to workflows.
*   **Custom Workflows:** Design tailored sequences of LLM calls for specific tasks.
*   **Multi-LLM Support:** Combine the strengths of multiple LLMs within a single workflow.
*   **Offline Wikipedia Integration:** Enhance factual accuracy using the Offline Wikipedia API.
*   **Continually Generated Chat Summaries:** Simulate conversational "memory" for long-running chats.
*   **Hotswap Models:** Maximize VRAM usage with Ollama's model hotswapping.
*   **Customizable Presets:** Easily configure LLM parameters with customizable JSON files.
*   **Vision Multi-Modal Support:** Experimental image processing via Ollama.
*   **Mid-Workflow Conditional Workflows:** Trigger specific workflows based on conditions.
*   **MCP Server Tool Integration:** Integrating MCP server tool calling allowing tool use mid-workflow.

## Getting Started

### 🛠️ Installation

1.  **Prerequisites:** Ensure Python is installed (3.10 or 3.12 recommended).
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run WilmerAI:**
    ```bash
    python server.py
    ```

    *   Alternatively, run the `.bat` (Windows) or `.sh` (macOS) scripts for convenient setup. See the original readme for script arguments.

### 📦 Pre-made Users: Fast Track Setup

Quickly set up WilmerAI using pre-configured user templates, including single-model and multi-model assistants, roleplay options, and coding workflows.  

*   **assistant-single-model**: A single small model used on all nodes, with category-specific presets.
*   **assistant-multi-model**: Utilize multiple models for different tasks with category-specific endpoints.
*   **convo-roleplay-single-model**: Good for conversations and roleplay.
*   **convo-roleplay-dual-model**: Leverages two models in tandem for high performance on conversation.
*   **group-chat-example**: Multi-persona group chats using different models.
*   **openwebui-norouting-single/dual-model**: Similar to roleplay models, but designed for Open WebUI.
*   **openwebui-routing-single/multi-model**: Similar to the assistant models, but designed for Open WebUI.
*   **socg-* (Advanced)**: Socg's personal coding and general workflows (designed for advanced users, see original readme).

#### Configuration

1.  **Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory.
2.  **Current User:** Set your current user in `Public/Configs/Users/_current-user.json`.
3.  **User Settings:** Customize your user configuration file (`Users/your_user.json`) with streaming preferences, Offline Wikipedia API settings, and file paths.
4.  **Routing:** In your User JSON specify what Routing file you would like to use, or use a pre-existing one.
5.  **Workflows:** In the Workflows folder, ensure you have a workflow that matches your user and each workflow from your routing file.

## Understanding Workflows

### ⚙️ Workflow Structure

Workflows are defined using JSON files and are located in the `Public/Workflows` directory within your user's specific workflows folder.

### 🧩 Workflow Nodes

Workflows consist of nodes that define interactions with LLMs.
-   **title**: for debugging
-   **agentName**: similar to title
-   **systemPrompt**: The system prompt to send to the LLM API.
-   **prompt**: The prompt to send. If left blank, either the last five messages from your conversation will be sent, or
  however many you specify.
-   **lastMessagesToSendInsteadOfPrompt**: Specify how many messages to send to the LLM if "prompt" is left as an empty
  string.
-   **endpointName**: The LLM API endpoint to send the prompt to. This should match a JSON file name from the `Endpoints`
  folder, without the `.json` extension.
-   **preset**: The preset to send to the API. Truncate length and max tokens to send come from this. This should match a
  JSON file name from the `Presets` folder, without the `.json` extension.
-   **maxResponseSizeInTokens**: Specifies the maximum number of tokens you want the LLM to send back to you as a
  response.
  This can be set per node, in case you want some nodes to respond with only 100 tokens and others to respond with 3000.
-   **addUserTurnTemplate**: Whether to wrap the prompt being sent to the LLM within a user turn template. If you send the
  last few messages, set this as `false` (see first example node above). If you send a prompt, set this as `true` (see
  second example node above).
-   **returnToUser**: This forces a node that is not the final node in a workflow to be the one to return its output
  to the user. This can be especially helpful in workflow lock scenarios. (please see
  the [Workflow lock section](#workflow-lock)). **IMPORTANT**: This only works for streaming workflows. This does not
  work for non-streaming.
-   **addDiscussionIdTimestampsForLLM**: This will generate timestamps and track them across your conversation starting
  from the moment you add this. The timestamps will be added to the beginning of any message sent to the LLM
  where that timestamp has been tracked. So, for example, if you turn this on after 10 messages have been sent, messages
  11 onward will be tracked on when the message arrived. When the messages are sent to the LLM, the timestamps will be
  included.
  
#### Node Types (examples)

*   **Recent Memory Summarizer Tool** - Summarize and use memories
*   **Full Chat Summary Node** - summarize an entire chat
*   **Python Module Caller Node** - Extend functionality with python
*   **Image Processor** - add images

### 🔐 Workflow Lock

Use workflow locks to avoid race conditions when performing asynchronous operations.  See details for the use case.

### 🖼️ Image Processor

Utilize an Ollama image endpoint in your workflows.

### ➡️ Conditional Custom Workflow Node

Dynamically execute sub-workflows based on conditions.

## Presets

Customize model parameters within JSON files.  See the original readme for more details on setup.

## Troubleshooting

See the original README for troubleshooting tips!

## 📧 Contact

WilmerAI.Project@gmail.com