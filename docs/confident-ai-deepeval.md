<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5917" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5917" alt="confident-ai%2Fdeepeval | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/3SEyvpgu2f?style=flat">
    </a>
</p>

<h4 align="center">
    <p>
        <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
        <a href="#-metrics-and-features">Metrics and Features</a> |
        <a href="#-quickstart">Getting Started</a> |
        <a href="#-integrations">Integrations</a> |
        <a href="https://confident-ai.com?utm_source=GitHub">DeepEval Platform</a>
    <p>
</h4>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/confident-ai/deepeval.svg?color=violet">
    </a>
    <a href="https://colab.research.google.com/drive/1PPxYEBa6eu__LquGoFFJZkhYgWVYE6kh?usp=sharing">
        <img alt="Try Quickstart in Colab" src="https://colab.research.google.com/assets/colab-badge.svg">
    </a>
    <a href="https://github.com/confident-ai/deepeval/blob/master/LICENSE.md">
        <img alt="License" src="https://img.shields.io/github/license/confident-ai/deepeval.svg?color=yellow">
    </a>
    <a href="https://x.com/deepeval">
        <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/deepeval?style=social&logo=x">
    </a>
</p>

<p align="center">
    <!-- Keep these links. Translations will automatically update with the README. -->
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=de">Deutsch</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=es">Español</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=fr">français</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ja">日本語</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ko">한국어</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=pt">Português</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=ru">Русский</a> | 
    <a href="https://www.readme-i18n.com/confident-ai/deepeval?lang=zh">中文</a>
</p>

## DeepEval: Effortlessly Evaluate and Enhance Your LLM Applications

DeepEval is an open-source framework designed to simplify and improve the evaluation of your large language model (LLM) applications, offering a suite of metrics, integrations, and a platform for comprehensive testing and analysis.  Explore the original repo on [GitHub](https://github.com/confident-ai/deepeval).

### Key Features:

*   **Comprehensive Metrics**: Evaluate LLM outputs with a wide array of ready-to-use metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG metrics: Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, and RAGAS.
    *   Agentic metrics: Task Completion, Tool Correctness
    *   Other metrics: Hallucination, Summarization, Bias, Toxicity
    *   Conversational metrics: Knowledge Retention, Conversation Completeness, Conversation Relevancy, and Role Adherence.
*   **Customization**: Build your own custom metrics and seamlessly integrate them into the DeepEval ecosystem.
*   **End-to-End & Component-Level Evaluation**: Supports both comprehensive end-to-end testing and granular component-level evaluations.
*   **Synthetic Dataset Generation**: Generate synthetic datasets to rigorously test and evaluate your LLM applications.
*   **CI/CD Integration**: Integrates effortlessly with any CI/CD environment for automated testing.
*   **Red Teaming Capabilities**: Red team your LLM application for over 40 safety vulnerabilities, including toxicity, bias, and prompt injection, using various attack strategies.
*   **LLM Benchmark Testing**: Easily benchmark any LLM on popular benchmarks like MMLU, HellaSwag, and more.
*   **Confident AI Integration**: Full integration with the [Confident AI](https://confident-ai.com?utm_source=GitHub) platform for:
    *   Dataset curation and annotation.
    *   LLM app benchmarking and comparison.
    *   Metric fine-tuning.
    *   Evaluation result debugging.
    *   Continuous monitoring and dataset improvement with real-world data.

### 🔥 Metrics and Features

> 🥳 You can now share DeepEval's test results on the cloud directly on [Confident AI](https://confident-ai.com?utm_source=GitHub)'s infrastructure

- Supports both end-to-end and component-level LLM evaluation.
- Large variety of ready-to-use LLM evaluation metrics (all with explanations) powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
  - G-Eval
  - DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
  - **RAG metrics:**
    - Answer Relevancy
    - Faithfulness
    - Contextual Recall
    - Contextual Precision
    - Contextual Relevancy
    - RAGAS
  - **Agentic metrics:**
    - Task Completion
    - Tool Correctness
  - **Others:**
    - Hallucination
    - Summarization
    - Bias
    - Toxicity
  - **Conversational metrics:**
    - Knowledge Retention
    - Conversation Completeness
    - Conversation Relevancy
    - Role Adherence
  - etc.
- Build your own custom metrics that are automatically integrated with DeepEval's ecosystem.
- Generate synthetic datasets for evaluation.
- Integrates seamlessly with **ANY** CI/CD environment.
- [Red team your LLM application](https://deepeval.com/docs/red-teaming-introduction) for 40+ safety vulnerabilities in a few lines of code, including:
  - Toxicity
  - Bias
  - SQL Injection
  - etc., using advanced 10+ attack enhancement strategies such as prompt injections.
- Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://deepeval.com/docs/benchmarks-introduction?utm_source=GitHub), which includes:
  - MMLU
  - HellaSwag
  - DROP
  - BIG-Bench Hard
  - TruthfulQA
  - HumanEval
  - GSM8K
- [100% integrated with Confident AI](https://confident-ai.com?utm_source=GitHub) for the full evaluation lifecycle:
  - Curate/annotate evaluation datasets on the cloud
  - Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
  - Fine-tune metrics for custom results
  - Debug evaluation results via LLM traces
  - Monitor & evaluate LLM responses in product to improve datasets with real-world data
  - Repeat until perfection

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

### 🔌 Integrations

*   🦄 [LlamaIndex](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 [Hugging Face](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

### 🚀 QuickStart

DeepEval provides an intuitive way to test and evaluate your LLM applications. Let's look at a sample test case:

#### Installation

```bash
pip install -U deepeval
```

#### Writing Your First Test Case

```python
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

def test_case():
    correctness_metric = GEval(
        name="Correctness",
        criteria="Determine if the 'actual output' is correct based on the 'expected output'.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        threshold=0.5
    )
    test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        # Replace this with the actual output from your LLM application
        actual_output="You have 30 days to get a full refund at no extra cost.",
        expected_output="We offer a 30-day full refund at no extra costs.",
        retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
    )
    assert_test(test_case, [correctness_metric])
```

#### Running the Test

```bash
export OPENAI_API_KEY="..."  # Set your OpenAI API key
deepeval test run test_chatbot.py
```

### LLM Evaluation with Confident AI

Unlock the full potential of DeepEval with the [Confident AI platform](https://confident-ai.com?utm_source=Github):

1.  Curate and annotate evaluation datasets.
2.  Benchmark LLM apps and compare iterations.
3.  Fine-tune metrics for precise results.
4.  Debug with LLM traces.
5.  Continuously improve datasets with real-world data.

Get started by logging in via the CLI:

```bash
deepeval login
```

### Contributing

We welcome contributions!  Please review our [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) to learn how to contribute.

### Roadmap

*   [x] Integration with Confident AI
*   [x] Implement G-Eval
*   [x] Implement RAG metrics
*   [x] Implement Conversational metrics
*   [x] Evaluation Dataset Creation
*   [x] Red-Teaming
*   [ ] DAG custom metrics
*   [ ] Guardrails

### Authors

DeepEval is built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

### License

DeepEval is licensed under Apache 2.0. See the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file.