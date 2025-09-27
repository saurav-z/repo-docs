<div align="center">
  <img src="https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/docs/assets/LazyLLM-logo.png" width="100%"/>
</div>

# LazyLLM: Build Powerful Multi-Agent LLM Applications with Low Code

**Tired of complex AI application development? LazyLLM offers a low-code solution for building and optimizing multi-agent LLM applications.**

[Original Repo](https://github.com/LazyAGI/LazyLLM) | [中文](README.CN.md) |  [EN](README.md)

[![CI](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml/badge.svg)](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub star chart](https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=flat-square)](https://star-history.com/#LazyAGI/LazyLLM)
[![](https://dcbadge.vercel.app/api/server/cDSrRycuM6?compact=true&style=flat)](https://discord.gg/cDSrRycuM6)

## Key Features of LazyLLM

*   🚀 **Low-Code Development:** Assemble complex AI applications with ease using pre-built modules and data flows, even without extensive LLM expertise.
*   ⚙️ **One-Click Deployment:** Simplify deployment of multi-agent applications, streamlining the process with a lightweight gateway mechanism.
*   🌐 **Cross-Platform Compatibility:** Deploy your applications across various infrastructures (bare-metal, development machines, Slurm clusters, public clouds) with one-click switching.
*   🔍 **Automated Parameter Optimization:** Efficiently tune your applications with grid search, optimizing base models, retrieval strategies, and fine-tuning parameters.
*   🏋️ **Efficient Model Fine-Tuning:** Fine-tune models directly within your applications to continuously improve performance, simplifying model iteration.

## Build AI Applications with LazyLLM

LazyLLM empowers you to create various AI applications quickly and efficiently. Here are a few examples:

### Chatbots

*   **Basic Chatbot:** Quickly set up a simple chatbot using a few lines of code:

    ```python
    # set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
    # or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
    import lazyllm
    chat = lazyllm.OnlineChatModule()
    lazyllm.WebModule(chat).start().wait()
    ```

*   **Advanced Multimodal Bot:** Build more sophisticated bots with multimodality and intent recognition, like the example in the original README, including image generation, speech recognition, and more.
  
### RAG (Retrieval-Augmented Generation)

*   Create powerful RAG applications for enhanced question answering, as shown in the original README, using both online and local deployments.

### Stories Creator

*   Generate stories with an outline writer and story generator with a few lines of code

### AI Painting Assistant

*   Quickly create an AI painting assistant using a few lines of code, turning prompts into stunning visuals.

## Core Capabilities

*   **Application Building:** Use pre-defined workflows like pipeline, parallel, diverter, if, switch, and loop to quickly create multi-agent AI applications. Support for one-click deployment and updates.
*   **Platform-Independent:** Experience a consistent user experience across various computing platforms.
*   **Model Support:** Fine-tuning and inference support for both local and online models, including popular frameworks and services like OpenAI and more.
*   **RAG Components:** Build retrieval-augmented generation applications with key components like Document, Parser, Retriever, and Reranker.
*   **Web Interfaces:** Includes basic web interfaces for chat and document management.

## Installation

### From Source Code

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

`pip install -r requirements.full.txt` is used when you want to finetune, deploy or build your rag application.

### From pip

Only install lazyllm and necessary dependencies, you can use:
```bash
pip3 install lazyllm
```

Install lazyllm and all dependencies, you can use:
```bash
pip3 install lazyllm
lazyllm install full
```

## Design Philosophy

LazyLLM is designed to address the challenges of large language models in production environments. The core approach is rapid prototyping, iterative optimization, and focus on algorithmic effectiveness to simplify the development process and handle the tedious engineering.

## Architecture

![Architecture](docs/assets/Architecture.en.png)

## Basic concepts

### Component

A Component is the smallest execution unit in LazyLLM; it can be either a function or a bash command. Components have three typical capabilities:
1. Cross-platform execution using a launcher, allowing seamless user experience:
  - EmptyLauncher: Runs locally, supporting development machines, bare metal, etc.
  - RemoteLauncher: Schedules execution on compute nodes, supporting Slurm, SenseCore, etc.
2. Implements a registration mechanism for grouping and quickly locating methods. Supports registration of functions and bash commands. Here is an example:

```python
import lazyllm
lazyllm.component_register.new_group('demo')

@lazyllm.component_register('demo')
def test(input):
    return f'input is {input}'

@lazyllm.component_register.cmd('demo')
def test_cmd(input):
    return f'echo input is {input}'

# >>> lazyllm.demo.test()(1)
# 'input is 1'
# >>> lazyllm.demo.test_cmd(launcher=launchers.slurm)(2)
# Command: srun -p pat_rd -N 1 --job-name=xf488db3 -n1 bash -c 'echo input is 2'
```

### Module

Modules are the top-level components in LazyLLM, equipped with four key capabilities: training, deployment, inference, and evaluation. Each module can choose to implement some or all of these capabilities, and each capability can be composed of one or more components. As shown in the table below, we have built-in some basic modules for everyone to use.

|      |Function | Training/Fine-tuning | Deployment | Inference | Evaluation |
| :---: | :---: | :---: | :---: | :---: | :---: |
| ActionModule | Can wrap functions, modules, flows, etc., into a Module | Supports training/fine-tuning of its Submodules through ActionModule | Supports deployment of its Submodules through ActionModule | ✅ | ✅ |
| UrlModule | Wraps any URL into a Module to access external services | ❌ | ❌ | ✅ | ✅ |
| ServerModule | Wraps any function, flow, or Module into an API service | ❌ | ✅ | ✅ | ✅
| TrainableModule | Trainable Module, all supported models are TrainableModules | ✅ | ✅ | ✅ | ✅ |
| WebModule | Launches a multi-round dialogue interface service | ❌ | ✅ | ❌ | ✅ |
| OnlineChatModule | Integrates online model fine-tuning and inference services | ✅ | ✅ | ✅ | ✅ |
| OnlineEmbeddingModule | Integrates online Embedding model inference services | ❌ | ✅ | ✅ | ✅ |


### Flow

Flow in LazyLLM defines the data stream, describing how data is passed from one callable object to another. You can use Flow to intuitively and efficiently organize and manage data flow. Based on various predefined Flows, we can easily build and manage complex applications using Modules, Components, Flows, or any callable objects. The Flows currently implemented in LazyLLM include Pipeline, Parallel, Diverter, Warp, IFS, Loop, etc., which can cover almost all application scenarios. Building applications with Flow offers the following advantages:
1. You can easily combine, add, and replace various modules and components; the design of Flow makes adding new features simple and facilitates collaboration between different modules and even projects.
2. Through a standardized interface and data flow mechanism, Flow reduces the repetitive work developers face when handling data transfer and transformation. Developers can focus more on core business logic, thus improving overall development efficiency.
3. Some Flows support asynchronous processing and parallel execution, significantly enhancing response speed and system performance when dealing with large-scale data or complex tasks.

## Future Plans

### Timeline
V0.6 Expected to start from September 1st, lasting 3 months, with continuous small version releases in between, such as v0.6.1, v0.6.2
V0.7 Expected to start from December 1st, lasting 3 months, with continuous small version releases in between, such as v0.7.1, v0.7.2

### Feature Modules
RAG
  - Engineering
    - Integrate LazyRAG capabilities into LazyLLM (V0.6)
    - Extend RAG's macro Q&A capabilities to multiple knowledge bases (V0.6)
    - RAG modules fully support horizontal scaling, supporting multi-machine deployment of RAG algorithm collaboration (V0.6)
    - Integrate at least 1 open-source knowledge graph framework (V0.6)
    - Support common data splitting strategies, no less than 20 types, covering various document types (V0.6)
  - Data Capabilities
    - Table parsing (V0.6 - 0.7)
    - CAD image parsing (V0.7 -)
  - Algorithm Capabilities
    - Support processing of relatively structured texts like CSV (V0.6)
    - Multi-hop retrieval (links in documents, references, etc.) (V0.6)
    - Information conflict handling (V0.7)
    - AgenticRL & code-writing problem-solving capabilities (V0.7)

Functional Modules
  - Support memory capabilities (V0.6)
  - Support for distributed Launcher (V0.7)
  - Database-based Globals support (V0.6)
  - ServerModule can be published as MCP service (v0.7)
  - Integration of online sandbox services (v0.7)

Model Training and Inference
  - Support OpenAI interface deployment and inference (V0.6)
  - Unify fine-tuning and inference prompts (V0.7)
  - Provide fine-tuning examples in Examples (V0.7)
  - Integrate 2-3 prompt repositories, allowing direct selection of prompts from prompt repositories (V0.6)
  - Support more intelligent model type judgment and inference framework selection, refactor and simplify auto-finetune framework selection logic (V0.6)
  - Full-chain GRPO support (V0.7)

Documentation
  - Complete API documentation, ensure every public interface has API documentation, with consistent documentation parameters and function parameters, and executable sample code (V0.6)
  - Complete CookBook documentation, increase cases to 50, with comparisons to LangChain/LlamaIndex (code volume, speed, extensibility) (V0.6)
  - Complete Environment documentation, supplement installation methods on win/linux/macos, supplement package splitting strategies (V0.6)
  - Complete Learn documentation, first teach how to use large models; then teach how to build agents; then teach how to use workflows; finally teach how to build RAG (V0.6)

Quality
  - Reduce CI time to within 10 minutes by mocking most modules (V0.6)
  - Add daily builds, put high-time-consuming/token tasks in daily builds (V0.6)

Development, Deployment and Release
  - Debug optimization (v0.7)
  - Process monitoring [output + performance] (v0.7)
  - Environment isolation and automatic environment setup for dependent training and inference frameworks (V0.6)

Ecosystem
  - Promote LazyCraft open source (V0.6)
  - Promote LazyRAG open source (V0.7)
  - Upload code to 2 code hosting websites other than Github and strive for community collaboration (V0.6)