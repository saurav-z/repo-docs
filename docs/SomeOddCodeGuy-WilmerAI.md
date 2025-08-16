# WilmerAI: Expertly Routing Your LLM Inference

**WilmerAI acts as a powerful intermediary, intelligently directing your prompts to multiple Language Models for enhanced results.** [Go to Original Repo](https://github.com/SomeOddCodeGuy/WilmerAI)

## Key Features

*   **Modular Prompt Routing:** Categorize and route prompts to custom workflows based on various criteria like domain (coding, math), or persona.
*   **Customizable Workflows:** Design intricate workflows to tailor how your prompts are processed, allowing for multi-LLM calls within a single request.
*   **Multi-LLM Orchestration:** Empower a single AI assistant with multiple LLMs operating concurrently within a workflow, maximizing quality.
*   **Offline Wikipedia Integration:** Utilize the [OfflineWikipediaTextApi](https://github.com/SomeOddCodeGuy/OfflineWikipediaTextApi) to enhance factual accuracy with Retrieval Augmented Generation (RAG).
*   **Persistent Conversation Summaries:** Generate continuous chat summaries to overcome LLM context limitations, sustaining coherent long-form conversations.
*   **Dynamic Model Hotswapping:** Optimize VRAM usage with Ollama's hotswapping, enabling complex workflows even on resource-constrained systems.
*   **Flexible Presets:** Configure LLM parameters through customizable JSON presets, adapting to new samplers or API requirements.
*   **Vision & Multimodal Support:** Experiment with image processing using Ollama, facilitating image analysis and interaction with compatible models.
*   **Mid-Workflow Branching:** Incorporate conditional workflows within existing workflows, triggering different paths based on LLM outputs.
*   **Tool Calling Integration (MCP):** Utilize server tool calling with the help of MCPO to enhance your workflow capabilities.

## What is WilmerAI?

WilmerAI is a tool designed to sit between your front end or application and the LLM APIs you are sending your prompts to.  From your perspective, it acts as a (likely long running) one-shot call to an LLM. But in reality,  it could be many LLMs and even tools doing complex work.

## Empowering Workflows

### Orchestrating Intelligence

Combine diverse LLMs, including those on older hardware, to construct sophisticated workflows. Distribute tasks across various models and APIs, streamlining performance and knowledge.

### Iterative Enhancement

Employ follow-up questions to refine LLM outputs. Build workflows that automate these steps, leading to higher quality responses, such as during coding tasks.

### Advanced Routing

WilmerAI moves beyond basic routing, enabling complex and adaptable categorizations, driven by custom workflows that empower users to steer LLM responses.

## Example Workflows (Illustrations)

[Include visual examples, as shown in the original README.]

## Setup & Configuration

[Include streamlined setup information, highlighting critical sections and potential arguments.]

## Connecting to WilmerAI

WilmerAI exposes OpenAI and Ollama compatible API endpoints for easy integration:

*   **OpenAI Compatibility:** `v1/completions` & `chat/completions`
*   **Ollama Compatibility:** `api/generate` & `api/chat`
*   **KoboldCpp Compatibility:** `api/v1/generate` & `/api/extra/generate/stream`

[Include detailed instructions for connecting via SillyTavern and Open WebUI.]

## Technical Notes & Disclaimers

[Briefly address maintainer's notes, the open source license, and important considerations around token usage.]

## Contact & Resources

*   **Email:** WilmerAI.Project@gmail.com
*   **Setup Tutorial:** [Include YouTube setup video link]
*   **Comprehensive Tutorial Playlist:** [Include YouTube tutorial playlist link]

## Third-Party Libraries

WilmerAI utilizes the following open-source libraries:

*   Flask
*   requests
*   scikit-learn
*   urllib3
*   jinja2

[Provide links to the license information.]

## License

WilmerAI is released under the GNU General Public License v3.