<div align="center">
  <img src="https://raw.githubusercontent.com/LazyAGI/LazyLLM/main/docs/assets/LazyLLM-logo.png" width="100%"/>
</div>

# LazyLLM: Build Powerful Multi-Agent LLM Applications with Low-Code Ease

Tired of complex LLM development? **LazyLLM is a low-code tool that streamlines the creation of multi-agent LLM applications, enabling rapid prototyping and iterative optimization.** ([Original Repo](https://github.com/LazyAGI/LazyLLM))

[![CI](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml/badge.svg)](https://github.com/LazyAGI/LazyLLM/actions/workflows/main.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellow.svg)](https://opensource.org/license/apache-2-0)
[![GitHub star chart](https://img.shields.io/github/stars/LazyAGI/LazyLLM?style=flat-square)](https://star-history.com/#LazyAGI/LazyLLM)
[![](https://dcbadge.vercel.app/api/server/cDSrRycuM6?compact=true&style=flat)](https://discord.gg/cDSrRycuM6)

## Key Features:

*   **Low-Code Development:** Build AI applications with multiple agents using a Lego-like assembly process, even without extensive LLM expertise.
*   **One-Click Deployment:** Simplify deployment of complex multi-agent applications, particularly during the Proof of Concept (POC) phase, with a lightweight gateway mechanism. Streamline image packaging for production.
*   **Cross-Platform Compatibility:** Seamlessly switch between IaaS platforms (bare-metal, development machines, Slurm clusters, public clouds) without code modifications, reducing migration efforts.
*   **Grid Search Parameter Optimization:** Automatically test different models, retrieval strategies, and parameters to efficiently optimize applications.
*   **Efficient Model Fine-Tuning:** Enhance application performance through in-application model fine-tuning, simplifying model iterations and accelerating algorithm development.

## What You Can Build

LazyLLM empowers you to create a variety of AI applications. Here are some examples:

### ChatBots

**Simple Chatbot Example:**

```python
# set environment variable: LAZYLLM_OPENAI_API_KEY=xx
# or you can make a config file(~/.lazyllm/config.json) and add openai_api_key=xx
import lazyllm
chat = lazyllm.OnlineChatModule()
lazyllm.WebModule(chat).start().wait()
```

**Advanced Bot Example (Multimodality & Intent Recognition):**

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

### RAG (Retrieval-Augmented Generation)

![Demo RAG](docs/assets/demo_rag.svg)

```python
import os
import lazyllm
from lazyllm import pipeline, parallel, bind, SentenceSplitter, Document, Retriever, Reranker

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'

# Online Deployment
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

```python
# Local Deployment
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

# Online Deployment
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.OnlineChatModule(stream=False).formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(lazyllm.OnlineChatModule(stream=False).prompt(writer_prompt))
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

```python
# Local Deployment
with pipeline() as ppl:
    ppl.outline_writer = lazyllm.TrainableModule('internlm2-chat-7b').formatter(JsonFormatter()).prompt(toc_prompt)
    ppl.story_generater = warp(ppl.outline_writer.share(prompt=writer_prompt).formatter())
    ppl.synthesizer = (lambda *storys, outlines: "\n".join([f"{o['title']}\n{s}" for s, o in zip(storys, outlines)])) | bind(outlines=ppl.output('outline_writer'))
lazyllm.WebModule(ppl, port=23466).start().wait()
```

### AI Painting Assistant

```python
import lazyllm
from lazyllm import pipeline

prompt = 'You are a drawing prompt word master who can convert any Chinese content entered by the user into English drawing prompt words. In this task, you need to convert any input content into English drawing prompt words, and you can enrich and expand the prompt word content.'

with pipeline() as ppl:
    ppl.llm = lazyllm.TrainableModule('internlm2-chat-7b').prompt(lazyllm.ChatPrompter(prompt))
    ppl.sd3 = lazyllm.TrainableModule('stable-diffusion-3-medium')
lazyllm.WebModule(ppl, port=23466).start().wait()
```

## Installation

### From Source Code:

```bash
git clone git@github.com:LazyAGI/LazyLLM.git
cd LazyLLM
pip install -r requirements.txt
```

For fine-tuning, deployment or RAG applications use:

```bash
pip install -r requirements.full.txt
```

### From PyPI:

Install only `lazyllm` and necessary dependencies:

```bash
pip3 install lazyllm
```

Install `lazyllm` and all dependencies:

```bash
pip3 install lazyllm
lazyllm install full
```

*(Rest of the original README content is excellent and can be included as-is after this point, but is too lengthy for summarization.)*