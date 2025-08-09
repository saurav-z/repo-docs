# WilmerAI: Expertly Routing Your LLM Inference

**Unlock the power of advanced workflows to orchestrate multiple Language Models (LLMs) and tools, delivering sophisticated AI-driven experiences.**  Dive into a world where complex prompts are handled by chains of LLMs and external resources. [View the Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features:

*   **Intelligent Prompt Routing:** Direct prompts to specific LLMs or custom categories based on content or persona.
*   **Customizable Workflows:** Craft unique multi-step processes that orchestrate multiple LLMs and even external tools.
*   **Multi-LLM Synergy:** Leverage multiple LLMs simultaneously to generate high-quality responses, overcoming single-model limitations.
*   **RAG Integration:** Utilize the [Offline Wikipedia API](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) to enrich responses with factual information.
*   **Conversation Summarization & Memory:** Maintain context in long conversations with continuously updated chat summaries and memory management.
*   **Hotswap Model Flexibility:** Optimize VRAM usage with Ollama's hotswapping capabilities, allowing use of multiple large models.
*   **Customizable Presets:** Fine-tune LLM behavior with user-friendly preset configurations.
*   **Vision & Multi-Modal Support:** Experiment with experimental vision and image processing through Ollama integration.
*   **Mid-Workflow Branching:** Dynamically create more complex flows with conditional and custom workflow nodes.
*   **MCPO Server Tool Integration:** Utilize external tools mid-workflow, a big thank you to [iSevenDays](https://github.com/iSevenDays) for the amazing work on this feature.
*   **And More!**

## Get Started

*   **Quick Setup:**  Utilize pre-built user configurations for rapid deployment or customize everything to fit your needs. Detailed instructions are available below to make your experience smooth.
*   **Comprehensive Tutorials:** Check out the setup tutorials on YouTube: [WilmerAI Setup Tutorial](https://www.youtube.com/watch?v=v2xYQCHZwJM) and [WilmerAI Tutorial Youtube PlayList](https://www.youtube.com/playlist?list=PLjIfeYFu5Pl7J7KGJqVmHM4HU56nByb4X)

## Understanding WilmerAI

WilmerAI acts as an intermediary, enhancing your interaction with Large Language Models. Unlike traditional LLM interfaces, it uses a highly configurable workflow system.

### Semi-Autonomous Workflows:

WilmerAI enables you to create workflows for complex problem solving:

### Key Use Cases:

*   **Complex Question Answering:** Combine multiple LLMs and tools (like a Wikipedia API) to provide accurate, comprehensive answers to difficult questions.
*   **Multi-Persona Chat:** Power group chat applications with different LLMs or AI personas by assigning each LLM to a persona or group to specialize their responses.
*   **Task Automation:** Design workflows for repetitive tasks (e.g., software development), improving accuracy and efficiency.

## Connecting to WilmerAI

WilmerAI exposes OpenAI and Ollama compatible API endpoints. This allows you to connect a range of frontend applications to it, giving you a flexible setup.

**Supported API Endpoints:**

*   OpenAI Compatible v1/completions (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate (*requires [Wilmer Prompt Template](Public/Configs/PromptTemplates/wilmerai.json)*)
*   Ollama Compatible api/chat

**Backend Connections:**

*   OpenAI Compatible v1/completions
*   OpenAI Compatible chat/completions
*   Ollama Compatible api/generate
*   Ollama Compatible api/chat
*   KoboldCpp Compatible api/v1/generate (*non-streaming generate*)
*   KoboldCpp Compatible /api/extra/generate/stream (*streaming generate*)

## Detailed Setup Guide

### Step 1: Installation

#### Option 1: Using Provided Scripts

*   **Windows:** Run the provided `.bat` file.
*   **macOS:** Run the provided `.sh` file.

#### Option 2: Manual Installation

1.  Install dependencies: `pip install -r requirements.txt`
2.  Run WilmerAI: `python server.py`

**Optional Arguments:**

*   `--ConfigDirectory`: Custom config files directory. Defaults to "Public".
*   `--LoggingDirectory`: Custom logs directory. File logging is OFF by default, "logs" if enabled.
*   `--User`: Specify the user profile.

### Step 2: Fast Route: Use Pre-made Users

Within the `Public/Configs` folder, find the following sub-folders, these are what you are going to be using:

*   `Endpoints`: Configure your LLM connections here, examples are provided.
*   `Users`:  Define and manage your user profiles.
*   `_current-user.json`: Set your active user.

**NOTE:** The Factual workflow nodes of the `assistant-single-model`, `assistant-multi-model`
and `group-chat-example` users will attempt to utilize the
[OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi)
project to pull full wikipedia articles to RAG against. If you don't have this API, the workflow
should not have any issues, but I personally use this API to help improve the factual responses I get.
You can specify the IP address to your API in the user json of your choice.

#### Recommended User Templates

1.  **assistant-single-model:** Ideal for workflows with a single LLM and multiple routes with different presets for each node.
2.  **assistant-multi-model:** Designed for workflows leveraging multiple LLMs, with each route utilizing different API settings, creating a more dynamic approach to problem solving.
3.  **convo-roleplay-single-model:** For a single model conversation.
4.  **convo-roleplay-dual-model:** For dual model conversation.
5.  **group-chat-example:** A detailed example of group chats.
6.  **socg-openwebui-norouting-coding-complex-multi-model:** Coding workflows that use different models that may take 20-30 minutes to perform complex tasks.
7.  **socg-openwebui-norouting-coding-dual-multi-model:** Coding workflows that rely on 2 models working in tandem to try to solve the issue. Much faster than the complex workflow, but still slower than asking single LLM.
8.  **socg-openwebui-norouting-coding-reasoning-multi-model:** Similar to the dual model coding workflow, but this expects a reasoning endpoint halfway through. Not a huge difference other than bigger max response sizes and a summary node before the responder.
9.  **socg-openwebui-norouting-coding-single-multi-model:** This calls a single coding model directly. Fastest response; not much different than just connecting directly to the LLM API, other than the support for the image endpoints.
10. **socg-openwebui-norouting-general-multi-model:** Calls a single general endpoint. This is good for things like RAG. Socg runs 3 variants of this: General, Large and Large-Rag (basically just copy/pasted the same user 3 times and changed the endpoints).
11. **socg-openwebui-norouting-general-offline-wikipedia:** This is similar to general-multi-model, but also makes a call to the  [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) to first pull an article.

### Step 3: Creating/Configuring a User

1.  **Update Endpoints:** Configure your LLM API endpoints in the `Public/Configs/Endpoints` directory. Adapt the sample files to your API settings.
2.  **Set Current User:** Set your active user in `Public/Configs/Users/_current-user.json`.
3.  **Review User Settings:** Modify the user settings within your new user's JSON file. Settings include streaming, API address, and offline wiki usage.

**That's it!** Run WilmerAI, connect your front end, and enjoy your new, powerful AI workflow tool.

### Step 4: Slow Route: Manual Setup of Endpoints, ApiTypes, PromptTemplates.

*   **Endpoints:** Address of the LLM API that you are connecting to.
    *   `apiTypeConfigFileName`: The exact name of the json file from the ApiTypes folder that specifies what type of API this is, minus the ".json" extension.
    *   `maxContextTokenSize`: Specifies the max token size that your endpoint can accept.
    *   `modelNameToSendToAPI`: Specifies what model name to send to the API.
    *   `promptTemplate`: The exact json file name of the prompt template to use, minus the ".json" extension.
    *   **and more!**
*   **ApiTypes:** These configuration files represent the different API types that you might be hitting when using Wilmer.
    *   `type`: Can be either: `KoboldCpp`, `OllamaApiChat`, `OllamaApiChatImageSpecific`, `OllamaApiGenerate`,
        `Open-AI-API`, `OpenAI-Compatible-Completions`, or `Text-Generation-WebUI`.
    *   `presetType`: This specifies the name of the folder that houses the presets you want to use.
    *   **and more!**
*   **PromptTemplates:**  Set templates by defining prefix, suffix, and end tokens for the system, user, and assistant roles.

### Step 5: Creating a User

1.  **Users Folder:** Create a user JSON file within the `Users` folder.
2.  **Update _current-user.json:** Set the new user's name in the `_current-user.json` file.
3.  **Routing Folder:** Create a routing configuration in the `Routing` folder.
4.  **Workflow Folder:** Create a workflow folder for your user in the `Workflow` directory, and copy files from another user, and/or the existing example workflows.

## Advanced Workflow Concepts

### Understanding Workflows

Workflows use nodes that send messages to an LLM endpoint.

*   **Node Properties:**  `title`, `agentName`, `systemPrompt`, `prompt`, `endpointName`, `preset`, `maxResponseSizeInTokens`, `addUserTurnTemplate`.

### Key Node Types:

*   **Recent Memory Summarizer Tool:** Summarize the conversation, add a memory tag to your conversation to use.
*   **Full Chat Summary Node:** Summarizes the entire chat based on the memories you have generated, add a discussion ID tag to your prompt, to use.
*   **Parallel Processing Node:**  Processes summaries or memories using multiple LLMs simultaneously.
*   **Python Module Caller Node:** Integrates custom Python scripts.
*   **OfflineWikiApiFullArticle Node:** Get the text to use in other node prompts, based on the  [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi)
*   **Image Processor Node:** Processes images through an Ollama API to then be used in another workflow.
*   **Get Custom File Node:** Grabs a custom file.
*   **Workflow Lock:** Stops the current workflow from processing so that you can continue chatting.
*   **Custom Workflow Node:** Allows you to call other workflows mid-way through the current one.
*   **Conditional Custom Workflow Node:** Dynamically selects sub-workflows based on certain conditions.

### Variables in Prompts

Use variables to make your prompts dynamic:

*   `{chat_user_prompt_last_one}`, `{chat_system_prompt}`, `{agent#Output}`, `{category_colon_descriptions}`, and more!

### Understanding Memories

> NOTE: For a deeper understanding of how memories work, please see
> the [Understanding Memories section](#understanding-memories)

The "Recent Memories" function is designed to enhance your conversation experience by chunking and summarizing your
messages, and then writing them to a specified text file.

**Enabling Memories:** Include a `[DiscussionId]#######[/DiscussionId]` tag in your system prompt or prompt.

### Understanding the Chat Summary

The "Chat Summary" function builds upon the "Recent Memories" by summarizing the entire conversation up to the current
point.

The chat summary is created via an internal loop within the `chatSummarySummarizer` type node.

**Enabling the Memory:** Include a tag in your conversation: `[DiscussionId]#######[/DiscussionId]`, where `######` is any
numerical value. You can insert this tag anywhere in the system
prompt or prompt; Wilmer should remove the tag before sending prompts to your LLM. Without this tag, the function
defaults to searching the last N number of messages instead.

### Parallel Processing

> NOTE: For a deeper understanding of how memories work, please see
> the [Understanding Memories section](#understanding-memories)

For handling extensive conversations, the app employs a parallel processing node for chat summaries and recent memories.
This allows you to distribute the workload across multiple LLMs. For example, if you have a conversation with 200,000
tokens resulting in about 200 memory chunks, you can assign these chunks to different LLMs. In a setup with three 8b
LLMs on separate computers, each LLM processes a chunk simultaneously, significantly reducing the processing time.

## Troubleshooting & Tips

*   **Front End Issues:** Ensure that "streaming" settings match between WilmerAI and your front end.
*   **Preset Compatibility:** Validate that your LLM supports the presets used.
*   **Error Diagnosis:** Review WilmerAI's output for clues about the causes of errors.
*   **Out-of-Memory/Truncation:** WilmerAI has no built-in token limits, make sure the LLM settings don't exceed the context of the models.
*   **User setup issues:** Double check that the settings exist.

## Contact & License

For questions, feedback, or contributions, contact WilmerAI.Project@gmail.com.

## Third Party Libraries

WilmerAI uses the following third-party libraries, further information can be found within the README of the ThirdParty-Licenses folder, as well as the full text of each license and their NOTICE files, if applicable, with relevant last updated dates for each.

*   Flask : https://github.com/pallets/flask/
*   requests: https://github.com/psf/requests/
*   scikit-learn: https://github.com/scikit-learn/scikit-learn/
*   urllib3: https://github.com/urllib3/urllib3/
*   jinja2: https://github.com/pallets/jinja

## License & Copyright

    WilmerAI
    Copyright (C) 2024 Christopher Smith

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.