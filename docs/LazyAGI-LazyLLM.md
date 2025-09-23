<div align="center">
  <img src="https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/docs/assets/LazyLLM-logo.png" width="100%"/>
</div>

# LazyLLM: Build Powerful Multi-Agent LLM Applications with Low-Code Ease

[LazyLLM](https://github.com/LazyAGI/LazyLLM) empowers developers to quickly build and iterate on sophisticated multi-agent LLM applications using a low-code approach.

[![CI](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml/badge.svg)](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub star chart](https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=flat-square)](https://star-history.com/#LazyAGI/LazyLLM)
[![](https://dcbadge.vercel.app/api/server/cDSrRycuM6?compact=true&style=flat)](https://discord.gg/cDSrRycuM6)

## Key Features

*   **Simplified Application Assembly:** Assemble AI applications with multiple agents easily, even without extensive LLM knowledge, using a Lego-like modular approach.
*   **One-Click Deployment:** Deploy complex multi-agent applications with a single click, streamlining the deployment process through a lightweight gateway.
*   **Cross-Platform Compatibility:** Easily switch between IaaS platforms (bare-metal, development machines, Slurm clusters, and public clouds) without code modifications.
*   **Automated Hyperparameter Optimization:** Leverage grid search for efficient model selection and parameter tuning.
*   **Efficient Fine-Tuning:** Fine-tune models within applications to continuously improve performance, simplifying model iteration and focusing on algorithm development.

## What You Can Build

LazyLLM accelerates the development of a wide range of AI applications. Here are a few examples.

### Chatbots

**This is a simple chatbot example.**

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat).start().wait()
```

If you want to use a locally deployed model, please ensure you have installed at least one inference framework (lightllm or vllm), and then use the following code

```python
import lazyllm
# Model will be downloaded automatically if you have an internet connection.
chat = lazyllm.TrainableModule('internlm2-chat-7b')
lazyllm.WebModule(chat, port=23466).start().wait()
```

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a chatbot by executing `lazyllm run chatbot`. If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a chatbot based on a local model by using `lazyllm run chatbot --model=internlm2-chat-7b`.

**Here's an advanced bot example with multimodality and intent recognition:**

![Demo Multimodal bot](docs/assets/multimodal-bot.svg)

<details>
<summary>click to look up prompts and imports</summary>

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline
from lazyllm.tools import IntentClassifier

painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'
```
</details>

```python
base = TrainableModule('internlm2-chat-7b')
with IntentClassifier(base) as ic:
    ic.case['Chat', base]
    ic.case['Speech Recognition', TrainableModule('SenseVoiceSmall')]
    ic.case['Image QA', TrainableModule('Mini-InternVL-Chat-2B-V1-5').deploy_method(deploy.LMDeploy)]
    ic.case['Drawing', pipeline(base.share().prompt(painter_prompt), TrainableModule('stable-diffusion-3-medium'))]
    ic.case['Generate Music', pipeline(base.share().prompt(musician_prompt), TrainableModule('musicgen-small'))]
    ic.case['Text to Speech', TrainableModule('ChatTTS')]
WebModule(ic, history=[base], audio=True, port=8847).start().wait()
```

### RAG (Retrieval-Augmented Generation)

![Demo RAG](docs/assets/demo_rag.svg)

<details>
<summary>click to look up prompts and imports</summary>

```python

import os
import lazyllm
from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'
```
</details>

Here is an online deployment example:

```python
documents = Document(dataset_path="your data path", embed=lazyllm.OnlineEmbeddingModule(), manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23466).start().wait()
```

Here is an example of a local deployment:

```python
documents = Document(dataset_path='/file/to/yourpath', embed=lazyllm.TrainableModule('bge-large-zh-v1.5'))
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)

with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3)

    ppl.reranker = Reranker("ModuleReranker", model="bge-reranker-large", topk=1) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str="".join([node.get_content() for node in nodes]), query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.TrainableModule("internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extra_keys=["context_str"]))

lazyllm.WebModule(ppl, port=23456).start().wait()
```

https://github.com/LazyAGI/LazyLLM/assets/12124621/77267adc-6e40-47b8-96a8-895df165b0ce

If you installed `lazyllm` using `pip` and ensured that the `bin` directory of your Python environment is in your `$PATH`, you can quickly start a retrieval-augmented bot by executing `lazyllm run rag --documents=/file/to/yourpath`. If you want to use a local model, you need to specify the model name with the `--model` parameter. For example, you can start a retrieval-augmented bot based on a local model by using `lazyllm run rag --documents=/file/to/yourpath --model=internlm2-chat-7b`.

### Stories Creator

<details>
<summary>click to look up prompts and imports</summary>

```python
import lazyllm
from lazyllm import pipeline, warp, bind
from lazyllm.components.formatter import JsonFormatter

toc_prompt="""
You are now an intelligent assistant. Your task is to understand the user's input and convert the outline into a list of nested dictionaries. Each dictionary contains a `title` and a `describe`, where the `title` should clearly indicate the level using Markdown format, and the `describe` is a description and writing guide for that section.

Please generate the corresponding list of nested dictionaries based on the following user input:

Example output:
[
    {
        "title": "# Level 1 Title",
        "describe": "Please provide a detailed description of the content under this title, offering background information and core viewpoints."
    },
    {
        "title": "## Level 2 Title",
        "describe": "Please provide a detailed description of the content under this title, giving specific details and examples to support the viewpoints of the Level 1 title."
    },
    {
        "title": "### Level 3 Title",
        "describe": "Please provide a detailed description of the content under this title, deeply analyzing and providing more details and data support."
    }
]
User input is as follows:
"""

completion_prompt="""
You are now an intelligent assistant. Your task is to receive a dictionary containing `title` and `describe`, and expand the writing according to the guidance in `describe`.

Input example:
{
    "title": "# Level 1 Title",
    "describe": "This is the description for writing."
}

Output:
This is the expanded content for writing.
Receive as follows:

"""

writer_prompt = {"system": completion_prompt, "user": '{"title": {title}, "describe": {describe}}'}
```
</details>

Here is an online deployment example:

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.OnlineChatModule(stream=False).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(lazyllm.OnlineChatModule(stream=False).prompt(writer_prompt))
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

Here is an example of a local deployment:

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

### AI Painting Assistant

<details>
<summary>click to look up prompts and imports</summary>

```python
import lazyllm
from lazyllm import pipeline

prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'
```
</details>

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## How LazyLLM Works

### Components

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

## Installation

### From Source Code

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

`pip install -r requirements.full.txt` is used when you want to finetune, deploy or build your rag application.

### From Pip

Only install lazyllm and necessary dependencies, you can use:
```bash
pip3 install lazyllm
```

Install lazyllm and all dependencies, you can use:
```bash
pip3 install lazyllm
lazyllm install full
```

## Join the Community

*   **Discord:**  [Discord Link](https://discord.gg/cDSrRycuM6)

## Future Plans

*   **RAG Enhancements:** Integration, multi-knowledge base support, horizontal scaling, knowledge graph integration, and improved data splitting strategies.
*   **Functional Modules:** Memory capabilities, distributed launcher support, database-based globals, and more.
*   **Model Training and Inference:** OpenAI interface, unified prompts, fine-tuning examples, and improved auto-finetune framework selection.
*   **Documentation:** Comprehensive API, cookbook, environment, and learn documentation.
*   **Quality and Development:** Reduced CI time, daily builds, debug optimization, and environment setup automation.
*   **Ecosystem:** Promotion of LazyCraft and LazyRAG, and wider code hosting.

---