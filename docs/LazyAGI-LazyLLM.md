<div align="center">
  <img src="https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/docs/assets/LazyLLM-logo.png" width="100%"/>
</div>

# LazyLLM: Build Multi-Agent LLM Applications with Ease

**Tired of complex AI application development?** LazyLLM is a low-code tool that simplifies building multi-agent LLM applications, empowering you to quickly prototype, optimize, and deploy AI solutions.  [View the original repository](https://github.com/LazyAGI/LazyLLM)

[![CI](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml/badge.svg)](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub star chart](https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=flat-square)](https://star-history.com/#LazyAGI/LazyLLM)
[![](https://dcbadge.vercel.app/api/server/cDSrRycuM6?compact=true&style=flat)](https://discord.gg/cDSrRycuM6)

## Key Features

*   **Low-Code Development:** Assemble AI applications with multiple agents using built-in data flows and functional modules, even without extensive LLM experience.
*   **Simplified Deployment:** Deploy complex applications with a single click, simplifying the process, especially during the POC (Proof of Concept) phase.
*   **Cross-Platform Compatibility:** Switch IaaS platforms seamlessly without code modifications, supporting bare-metal servers, development machines, Slurm clusters, and public clouds.
*   **Automated Parameter Optimization:**  Utilize grid search to find the best configurations without extensive code changes.
*   **Efficient Fine-Tuning:** Fine-tune models directly within applications to enhance performance, focusing on algorithm and data iteration.

## What Can You Build with LazyLLM?

LazyLLM empowers you to create various AI applications, including:

### Chatbots

**Simple Chatbot Example:**

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx 
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat).start().wait()
```

**Advanced Multimodal Bot Example:**  *Combining multiple functionalities such as Chat, Speech Recognition, Image QA, Drawing, etc.*

![Demo Multimodal bot](docs/assets/multimodal-bot.svg)

```python
from lazyllm import TrainableModule, WebModule, deploy, pipeline
from lazyllm.tools import IntentClassifier

painter_prompt = 'Now you are a master of drawing prompts, capable of converting any Chinese content entered by the user into English drawing prompts. In this task, you need to convert any input content into English drawing prompts, and you can enrich and expand the prompt content.'
musician_prompt = 'Now you are a master of music composition prompts, capable of converting any Chinese content entered by the user into English music composition prompts. In this task, you need to convert any input content into English music composition prompts, and you can enrich and expand the prompt content.'

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

### Retrieval-Augmented Generation (RAG)

![Demo RAG](docs/assets/demo_rag.svg)

**Online Deployment Example:**

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

**Local Deployment Example:**

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

### Story Creator

**Online Deployment Example:**

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.OnlineChatModule(stream=False).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(lazyllm.OnlineChatModule(stream=False).prompt(writer_prompt))
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

**Local Deployment Example:**

```python
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

### AI Painting Assistant

```python
with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## Getting Started

### Installation

**From Source Code:**

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

**From Pip:**

```bash
pip3 install lazyllm
```

To install with all dependencies needed for features like finetuning and deployment, use:

```bash
pip3 install lazyllm
lazyllm install full
```

## Core Concepts

LazyLLM is built upon these key concepts:

*   **Component:**  The smallest executable unit, allowing cross-platform execution.
*   **Module:**  Top-level components with training, deployment, inference, and evaluation capabilities. Pre-built modules simplify common tasks.
*   **Flow:** Defines data streams to organize and manage data flow within applications.

## Future Plans

*   **RAG Enhancements:** Integrate LazyRAG capabilities and support for multiple knowledge bases.
*   **Functional Modules:**  Add memory capabilities, distributed launcher support, and more.
*   **Model Training and Inference:**  Support OpenAI interface deployment and improve fine-tuning.
*   **Comprehensive Documentation:** Complete API documentation, CookBook documentation, and environment setup guides.
*   **Quality and Development:**  Reduce CI time and improve debug features.