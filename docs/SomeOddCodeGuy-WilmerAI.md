# WilmerAI: Unleash the Power of Multi-LLM Workflows (SEO-Optimized)

**Tired of limited AI responses?** WilmerAI is a flexible middleware application that connects your front-end application to a network of Large Language Models (LLMs), offering unparalleled control over your AI interactions and enabling complex, multi-step workflows. [Explore the WilmerAI Project](https://github.com/SomeOddCodeGuy/WilmerAI) for advanced AI orchestration!

## Key Features of WilmerAI:

*   **Multi-LLM Orchestration**: Leverage multiple LLMs (OpenAI, Ollama, KoboldCpp, etc.) in a single workflow for enhanced performance and diverse outputs.
*   **Advanced Prompt Routing**: Route prompts to custom categories or personas, allowing you to organize and control the AI's responses.
*   **Customizable Workflows**: Build complex, iterative workflows that automate multi-step tasks and tailor responses to your specific needs.
*   **Enhanced Context Management**: Utilize iterative LLM calls and configurable memory systems that help maintain consistency in long-running conversations that exceed the LLM's default context.
*   **RAG Support (Offline Wikipedia API)**: Integrate with the [Offline Wikipedia API](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) for Retrieval-Augmented Generation.
*   **Image Processing Support**: Experiment with image processing via Ollama.
*   **Modular Presets**: Customize model behavior easily with modifiable presets for various LLMs.
*   **Workflow Locks**: Utilize workflow locks to allow complex multi-step conversations without stalling.
*   **Tool Use via iSevenDays' MCP Tools**: Utilizing MCPO, Wilmer can now handle tool use mid-workflow. (experimental)

## Getting Started with WilmerAI

WilmerAI is designed for flexibility and control. Whether you're a seasoned developer or new to AI, this guide will give
you the tools you need to utilize its advanced capabilities.

### **1. Setup**

#### **Installation**

To install WilmerAI, please follow these steps.

**Option 1: Using Provided Scripts**
*   To use the provided scripts, first ensure that you have python installed, then run these scripts.
*   For Windows, run the provided `.bat` file.
*   For MacOS, run the provided `.sh` file.
*   For Linux: You can adapt the .sh file if you want, but none is provided as the author doesn't have a linux machine.

>   **IMPORTANT:** Always inspect `.bat` and `.sh` files before running them.

**Option 2: Manual Installation**
1.  Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
2.  Start the program:
    ```bash
    python server.py
    ```

#### **Configuration**

The main configurations files can be found in the `Public` folder. When updating WilmerAI, you should simply copy your
"Public" folder to retain your settings.

This section walks you through setting up Wilmer.

**IMPORTANT NOTES**

> *   A) Preset files are 100% customizable. What is in that file goes to the LLM API. This is because cloud APIs do not
>     handle some of the various presets that local LLM APIs handle. As such, if you use OpenAI API or other cloud
>     services, the calls will probably fail if you use one of the regular local AI presets. Please see the preset
>     "OpenAI-API" for an example of what openAI accepts.
> *   B) I have recently replaced all prompts in Wilmer to go from using the second person to third person. This has
>     had pretty decent results for me, and I'm hoping it will for you as well.
> *   C) By default, all the user files are set to turn on streaming responses. You either need to enable this in your
>     front end that is calling Wilmer so that both match, or you need to go into Users/username.json and set Stream to
>     "false". If you have a mismatch, where the front end does/does not expect streaming and your wilmer expects the
>     opposite, nothing will likely show on the front end.

##### **Step 1: Choose/Create Your User**

1.  **Fast Route**: Choose an example user from the `Public/Configs/Users` directory. This folder contains a series of
    JSON files, each representing a user configuration. The easiest way to get started is to copy an existing user and
    modify it to your needs.
    *   **`assistant-single-model`**: For a single LLM for various categories.
    *   **`assistant-multi-model`**: Uses multiple LLMs in tandem, with each category having its own endpoint.
    *   **`convo-roleplay-single-model`**: A simple conversation user that bypasses routing.
    *   **`convo-roleplay-dual-model`**: Uses two models, one for responding and another for summarizing.
    *   **`group-chat-example`**: Example of group chats using actual characters (SillyTavern compatible).
    *   **`openwebui-norouting-single-model`**: Adapted for Open WebUI.
    *   **`openwebui-norouting-dual-model`**: Adapted for Open WebUI.
    *   **`openwebui-routing-multi-model`**: Adapted for Open WebUI.
    *   **`openwebui-routing-single-model`**: Adapted for Open WebUI.
    *   **`socg-openwebui-norouting-coding-complex-multi-model`**: (Socg Coding Example) A complex coding workflow.
    *   **`socg-openwebui-norouting-coding-dual-multi-model`**: (Socg Coding Example) Dual Model Coding.
    *   **`socg-openwebui-norouting-coding-reasoning-multi-model`**: (Socg Coding Example) Coding workflow with reasoning.
    *   **`socg-openwebui-norouting-coding-single-multi-model`**: (Socg Coding Example) Single Model Coding.
    *   **`socg-openwebui-norouting-general-multi-model`**: (Socg General Example) General workflow for tasks.
    *   **`socg-openwebui-norouting-general-offline-wikipedia`**: (Socg General Example) Utilizes Offline Wikipedia API.
    >   NOTE: You may need to update the Endpoints for your own models.

2.  **Slow Route**: You can also build your own user configuration. Create a JSON file in the `Users` folder. The
    essential properties are:
    *   `port`: Port number for WilmerAI.
    *   `stream`: Whether to stream responses.
    *   `customWorkflowOverride`: Allows directing all prompts to a specific workflow.
    *   `customWorkflow`: Workflow file to use if `customWorkflowOverride` is true.
    *   `routingConfig`: Name of the routing config.
    *   `categorizationWorkflow`: Workflow for prompt categorization.

##### **Step 2: Configure Endpoints**

Endpoints are defined in the `Public/Configs/Endpoints` folder. These files specify the LLM API details.
1.  Update the endpoints for your user under Public/Configs/Endpoints.

    *   **`endpoint`**: Address for connecting to the LLM API. (OpenAI compatible)
    *   **`apiTypeConfigFileName`**: Specifies the name of the file from the `ApiTypes` folder.
    *   **`maxContextTokenSize`**: The maximum number of tokens the endpoint can accept.
    *   **`modelNameToSendToAPI`**: Model name to send to the API.

##### **Step 3: Set Your Current User**

Specify which user you want to use. You can do this using the `--User` argument during the script run (recommended for
multiple instances). Or, edit the `_current-user.json` in the `Public/Configs/Users` folder, with the name of the user.

##### **Step 4: Routing Setup**

Create or use an existing routing configuration file in the `Public/Configs/Routing` folder.

The core routing structure is as follows:

```json
{
  "CODING": {
    "description": "Any request which requires a code snippet as a response",
    "workflow": "CodingWorkflow"
  },
  "FACTUAL": {
    "description": "Requests that require factual information or data",
    "workflow": "ConversationalWorkflow"
  },
  "CONVERSATIONAL": {
    "description": "Casual conversation or non-specific inquiries",
    "workflow": "FactualWorkflow"
  }
}
```

*   **Element Name**: Category like CODING, FACTUAL, etc.
*   **description**: Descriptive text sent to the categorizing LLM.
*   **workflow**: The workflow file to run if this category is chosen.

##### **Step 5: Workflows Setup**

Workflows can be found in the `Public/Workflows` folder.

Within your user's specific workflow folder, create JSON files to define the workflows. The core concept is
nodes that run sequentially.

1.  Create a `Workflows` folder
2.  Create a directory inside of `Workflows` with your username as the directory name.
3.  Copy and rename your workflow files into this directory.

### 2. Connecting to WilmerAI

#### Supported APIs

WilmerAI exposes a number of APIs that will allow you to connect most LLM front-ends.

*   OpenAI Compatible `v1/completions`
*   OpenAI Compatible `chat/completions`
*   Ollama Compatible `api/generate`
*   Ollama Compatible `api/chat`
*   KoboldCpp Compatible `api/v1/generate` (non-streaming)
*   KoboldCpp Compatible `/api/extra/generate/stream` (streaming)

#### Integrating with SillyTavern

To connect SillyTavern, set up the connections as follows:

**Text Completion**
Connect as OpenAI Compatible v1/Completions:

Connect as Ollama api/generate:

**Chat Completions (not recommended)**

To connect as Open AI Chat Completions in SillyTavern, follow these steps:

*   Once connected, your presets are largely irrelevant and will be controlled by Wilmer; settings like temperature,
    top\_k, etc. The only field you need to update is your truncate length. I recommend setting it to the maximum your
    front end will allow; in SillyTavern, that is around 200,000 tokens.
*   If you connect via chat/Completion, please go to presets, expand "Character Names Behavior", and set it to "Message
    Content". If you do not do this, then go to your Wilmer user file and set `chatCompleteAddUserAssistant` to true. (I
    don't recommend setting both to true at the same time. Do either character names from SillyTavern, OR user/assistant
    from Wilmer. The AI might get confused otherwise.)

#### Integrating with Open WebUI

When connecting to Wilmer from Open WebUI, simply connect to it as if it were an Ollama instance.

### 3. Understanding Workflows

WilmerAI's core strength is its workflow engine. Workflows determine the path your prompts take.

#### **Workflow Structure**

Workflows are JSON files made up of "nodes" that execute sequentially. The system has been updated to support a more
powerful dictionary-based format that allows for top-level configuration and variables, making workflows much cleaner
and easier to manage.

#### **New Format (Recommended)**

This format allows you to define custom variables at the top level of the JSON. These variables can then be used in any
node throughout the workflow.

```json
{
  "persona": "You are a helpful and creative AI assistant.",
  "shared_endpoint": "OpenWebUI-NoRouting-Single-Model-Endpoint",
  "nodes": [
    {
      "title": "Gather Relevant Memories",
      "type": "VectorMemorySearch",
      "endpointName": "{shared_endpoint}"
    },
    {
      "title": "Respond to User",
      "type": "Standard",
      "systemPrompt": "{persona}\n\nHere are some relevant memories from our past conversations:\n[\n{agent1Output}\n]",
      "endpointName": "{shared_endpoint}",
      "preset": "Conversational_Preset",
      "returnToUser": true
    }
  ]
}
```

#### Node Properties

*   `type` **(Required)**: Defines node function.
*   `title`: Node name.
*   `systemPrompt`: System prompt.
*   `prompt`: User prompt.
*   `lastMessagesToSendInsteadOfPrompt`: Recent messages to include.
*   `endpointName`: LLM API endpoint.
*   `preset`: Preset from `Presets` folder.
*   `maxResponseSizeInTokens`: Override the preset.
*   `addUserTurnTemplate`: Wraps prompt with user template.
*   `returnToUser`: Output to user immediately.
*   `workflowName`: Sub-workflow file for a `"CustomWorkflow"` node.
*   `scoped_variables`: Values to pass into a custom workflow.

#### Variables in Prompts

*   `{agent#Output}`: Output from a previous node in the same workflow.
*   `{agent#Input}`: Variable from the parent workflow.
*   `{custom_variable}`: Custom variable defined at the workflow's top level.
*   Conversation & Message Variables (e.g., `{chat_user_prompt_last_one}`).
*   Date, Time & Context Variables (e.g., `{todays_date_iso}`).
*   Prompt Routing Variables (e.g., `{category_colon_descriptions}`).
*   Special Variables (e.g., `[TextChunk]`).

#### Workflow Nodes

*   "QualityMemory" node
*   "FullChatSummary" node
*   "VectorMemorySearch" node
*   "PythonModule" node
*   "ConditionalCustomWorkflow" node
*   "GetCustomFile" node
*   "ImageProcessor" node
*   "WorkflowLock" node
*   "OfflineWikiApiFullArticle" node

## Troubleshooting

*   **No memories or summaries**: Verify correct nodes exist, and that the directory in Users/\<username>.json is correct.
*   **No response on front-end**: Verify streaming is enabled/disabled to match.
*   **Error related to presets**: Check to ensure the preset matches what the LLM API accepts.
*   **Out of memory/truncate errors**: Review model's maximum token count.
*   **"None type" errors**: Check LLM API availability.

## Contact

For questions and feedback, contact WilmerAI.Project@gmail.com.

## Third-Party Libraries

WilmerAI uses the following libraries.

*   Flask
*   requests
*   scikit-learn
*   urllib3
*   jinja2
*   pillow

## License

WilmerAI is licensed under the GNU General Public License, Version 3.