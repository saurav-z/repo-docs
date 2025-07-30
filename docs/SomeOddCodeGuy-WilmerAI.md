# WilmerAI: Expertly Routing Language Models for Powerful AI Workflows

**Unlock the potential of multiple language models working together with WilmerAI, a versatile middleware designed to orchestrate and enhance your AI interactions.** ([Back to Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI))

WilmerAI sits between your LLM APIs and applications like Open WebUI or SillyTavern, enabling you to harness the power of diverse models for a single, comprehensive response.

## Key Features:

*   **Prompt Routing:** Direct prompts to various categories (coding, math, personas) with custom presets.
*   **Custom Workflows:** Design and control the flow of prompts through multiple LLMs.
*   **Multi-LLM Responses:** Leverage multiple models simultaneously for a single response.
*   **RAG Support:** Seamless integration with the Offline Wikipedia API ([OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi)) to enhance factual accuracy.
*   **Persistent Chat Summaries:**  Generate "memories" by chunking, summarizing, and constantly updating conversation summaries for enhanced context.
*   **Model Hotswapping:** Utilize Ollama's hotswapping feature to maximize VRAM usage and run complex workflows on limited hardware.
*   **Customizable Presets:** Easily configure LLM parameters using customizable JSON files.
*   **Vision Multi-Modal Support (Ollama):** Experimental image processing when using Ollama, to query LLMs about multiple images.
*   **Mid-Workflow Logic:**  Implement conditional workflows based on LLM responses for dynamic processing.
*   **MCP Server Tool Integration (MCPO):** Experimental support for MCP server tool calling with MCPO, allowing tool use mid-workflow.

## What Can You Do with WilmerAI?

WilmerAI empowers you to create sophisticated AI workflows:

*   **RAG-Enhanced Responses:** Easily integrate Retrieval-Augmented Generation (RAG) to ground responses with information.

*   **Iterative LLM Refinement:** Build workflows that utilize follow-up questions to greatly improve performance.

*   **Distributed LLM Processing:** Utilize multiple LLMs across various machines for powerful AI assistance.

## Getting Started

### Installation

1.  **Prerequisites:** Ensure you have Python installed (3.10 or 3.12 recommended).
2.  **Choose an Installation Method:**
    *   **Provided Scripts (Recommended):**  Use the `.bat` (Windows) or `.sh` (macOS) file to create a virtual environment and install dependencies.
    *   **Manual Installation:**  
        ```bash
        pip install -r requirements.txt
        python server.py
        ```
3.  **Configuration:** All configurations are managed through JSON files in the `Public` folder.

### Step-by-Step Setup

1.  **Endpoints:** Configure your LLM API endpoints in `Public/Configs/Endpoints`.  See the "Endpoints" section below for detailed instructions.
2.  **Users:** Create a user configuration file in `Public/Configs/Users` (copy an existing one and rename it, it's the easiest method).
3.  **Current User:** Set the `_current-user.json` file in `Public/Configs/Users` to your user's name.
4.  **Routing:** Define prompt routing rules in `Public/Configs/Routing`.
5.  **Workflows:** Create a new folder matching your username within `Public/Workflows` and create your workflow .json files.  Adjust settings to your specifications.

## Important Considerations

*   **Token Tracking:** WilmerAI does not currently track token usage.  Monitor your LLM API dashboards for accurate token consumption.
*   **Model Quality:** The quality of your LLMs, presets, and prompt templates directly impacts WilmerAI's output.
*   **Streaming Control:** Ensure streaming settings match between WilmerAI and your front end.

## Key Concepts

### Endpoints

Configure the connection details for your LLM APIs in the `Endpoints` directory (JSON files).

- `endpoint`: LLM API address.
- `apiTypeConfigFileName`: API type from the ApiTypes folder.
- `maxContextTokenSize`: Maximum context size.
- `modelNameToSendToAPI`: The Model name to send to API.
- `promptTemplate`: Prompt template file name.
- `removeThinking`: Removes "thinking" tags, which is helpful with reasoning models.

### Prompt Templates

Define prompt formats in the `PromptTemplates` directory.

### ApiTypes

Configure different API types to support the various different endpoints.

### Users

Set up various types of users with settings that control how the prompts are routed and processed in the `Users` directory.

### Workflows

Design multi-step processes in the `Workflows` directory, composed of node-based processing using JSON files.

### Understanding Workflow Nodes

-   **`title`:**  Internal name for debugging.
-   **`agentName`:**  Node name (for tracking output).
-   **`systemPrompt`:** The System prompt for the LLM API.
-   **`prompt`:** The actual user prompt.
-   **`lastMessagesToSendInsteadOfPrompt`:** Number of messages to send.
-   **`endpointName`:**  The connected endpoint.
-   **`preset`:** Preset file for LLM settings.
-   **`maxResponseSizeInTokens`:** Token limit for the response.
-   **`addUserTurnTemplate`:** Template usage boolean.
-   **`returnToUser`**: Forces the node to return its output. Useful for locks.

### Variables in Prompts

*   `{chat_user_prompt_last_one}`: Last message in the conversation
*   `{templated_user_prompt_last_one}`: Last message in conversation, templated.
*   `{chat_system_prompt}`: System Prompt
*   `{templated_system_prompt}`: System prompt with template applied.
*   `{agent#Output}`: Output from a previous agent.
*   `{category_colon_descriptions}`: The list of Categories and their Descriptions.
*   `{category_colon_descriptions_newline_bulletpoint}`: The list of Categories and their Descriptions, bulleted list.
*   `{categoriesSeparatedByOr}`: Category names, separated by "OR".
*   `{categoryNameBulletpoints}`: Bulleted list of category names.
*   `[TextChunk]`: Special variable unique to the parallel processor.

### Nodes:

**Get Custom File**
*   `filepath`: The file path to load.
*   `delimiter`: The delimiter to separate items in the file.
*   `customReturnDelimiter`: The delimiter to replace in the output.

**Recent Memory Summarizer Tool**
*   `maxTurnsToPull`
*   `maxSummaryChunksFromFile`
*   `lookbackStart`
*   `customDelimiter`

**Recent/Quality Memory Node**
*   `type: "QualityMemory"`

**Full Chat Summary Node**
*   `loopIfMemoriesExceed`
*   `minMemoriesPerSummary`

**Get Current Chat Summary From File**
*   `type: "GetCurrentSummaryFromFile"`

**Parallel Processing Node**
*   `multiModelList`
*   `preset`
*   `prompt`
*   `systemPrompt`
*   `ragTarget`
*   `ragType`

**Python Module Caller Node**
*   `module_path`
*   `args`
*   `kwargs`
*   `type: "PythonModule"`

**Full Text Wikipedia Offline API Caller Node**
*   `promptToSearch`
*   `type: "OfflineWikiApiFullArticle"`

**First Paragraph Text Wikipedia Offline API Caller Node**
*   `promptToSearch`
*   `type: "OfflineWikiApiPartialArticle"`

**Workflow Lock**
*   `type: "WorkflowLock"`
*   `workflowLockId`

**Image Processor**
*   `addAsUserMessage`: Add the image caption as a new user message.
*   `message`: The message to add.  Includes `[IMAGE_BLOCK]` for the caption.

**Custom Workflow Node**
*   `workflowName`
*   `is_responder`
*   `firstNodeSystemPromptOverride`
*   `firstNodePromptOverride`

**Conditional Custom Workflow Node**
*   `conditionalKey`: key for conditions (e.g., agent output).
*   `conditionalWorkflows`: Map `conditionalKey` to workflows.
    *   `Default`: Fallback workflow.
*   `is_responder`:  Flag to finish workflow.
*   `routeOverrides`: Overrides system or user prompts per route.

---

## Quick Troubleshooting Tips

*   **No Memories/Summary:** Ensure the file exists, the nodes are in your workflow, and the memory folder is specified correctly.
*   **No Response:** Verify streaming settings match between WilmerAI and the front end.
*   **Preset Errors:** Ensure your preset is compatible with your LLM API.
*   **OOM Errors:** WilmerAI has no token limits, but you might want to include them.
*   **"NoneType" Errors:** Check logs and verify API responses.
*   **Stalled Prompts:** Confirm the endpoint's address, port, and user settings.

---

## Contact

For support, suggestions, or just to say hello, please reach out to:

WilmerAI.Project@gmail.com

---

## Third-Party Libraries

WilmerAI uses the following libraries:

*   Flask: [https://github.com/pallets/flask/](https://github.com/pallets/flask/)
*   requests: [https://github.com/psf/requests/](https://github.com/psf/requests/)
*   scikit-learn: [https://github.com/scikit-learn/scikit-learn/](https://github.com/scikit-learn/scikit-learn/)
*   urllib3: [https://github.com/urllib3/urllib3/](https://github.com/urllib3/urllib3/)
*   jinja2: [https://github.com/pallets/jinja](https://github.com/pallets/jinja)

For details on licensing, please refer to the ThirdParty-Licenses folder.

## License
WilmerAI is licensed under the GNU General Public License v3.0. See the LICENSE file for details.