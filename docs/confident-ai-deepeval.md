<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>

<p align="center">
  <a href="https://github.com/confident-ai/deepeval">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social">
  </a>
  <a href="https://discord.gg/3SEyvpgu2f">
    <img alt="Discord" src="https://img.shields.io/discord/1123255868984291900?logo=discord&label=Discord">
  </a>
  <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">
      <img alt="Documentation" src="https://img.shields.io/badge/documentation-deepeval.com-blue">
  </a>
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
  <a href="https://trendshift.io/repositories/5917" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5917" alt="confident-ai%2Fdeepeval | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
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

---

**DeepEval empowers you to thoroughly evaluate and test your LLM applications with ease, just like unit testing.**

DeepEval is an open-source LLM evaluation framework designed for comprehensive testing and evaluation of large language model systems.  It offers a powerful suite of tools similar to Pytest, but specifically tailored for unit testing and evaluating LLM outputs.

## Key Features

*   **Versatile Evaluation**: Supports both end-to-end and component-level LLM evaluation, giving you granular control over testing.
*   **Extensive Metric Library**:  Provides a wide array of ready-to-use LLM evaluation metrics (with detailed explanations), including:
    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG Metrics**: Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS
    *   **Agentic Metrics**: Task Completion, Tool Correctness
    *   **Additional Metrics**: Hallucination, Summarization, Bias, Toxicity, Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence
    *   And more!
*   **Custom Metric Creation**: Build and seamlessly integrate your own custom metrics into the DeepEval ecosystem.
*   **Synthetic Dataset Generation**: Generate synthetic datasets for effective evaluation of your models.
*   **CI/CD Integration**: Integrates seamlessly with any CI/CD environment for automated testing and deployment.
*   **Red Teaming Capabilities**: Red team your LLM application for 40+ safety vulnerabilities.
*   **Benchmarking**: Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://deepeval.com/docs/benchmarks-introduction?utm_source=GitHub), which includes: MMLU, HellaSwag, DROP, BIG-Bench Hard, TruthfulQA, HumanEval, GSM8K
*   **Confident AI Integration**:  100% integrated with the DeepEval platform [Confident AI](https://confident-ai.com?utm_source=GitHub) for the full evaluation lifecycle:
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

## Getting Started

### Installation

```bash
pip install -U deepeval
```

###  DeepEval Login
Using the `deepeval` platform will allow you to generate sharable testing reports on the cloud. It is free, takes no additional code to setup, and we highly recommend giving it a try.
```bash
deepeval login
```

### Creating a Test Case

Example:

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

### Running Your Tests

```bash
export OPENAI_API_KEY="..." # Set your OpenAI API Key or custom model API key
deepeval test run test_chatbot.py
```

For more detailed information and advanced usage, explore the official [DeepEval Documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub).

## Integrations

*   🦄 [LlamaIndex](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 [Hugging Face](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

## Contributing

Contribute to DeepEval!  Read the [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) to learn how to submit pull requests and contribute to the project.

## Roadmap

*   [x] Integration with Confident AI
*   [x] Implement G-Eval
*   [x] Implement RAG metrics
*   [x] Implement Conversational metrics
*   [x] Evaluation Dataset Creation
*   [x] Red-Teaming
*   [ ] DAG custom metrics
*   [ ] Guardrails

## Authors

DeepEval is built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

## License

DeepEval is licensed under the Apache 2.0 license. See [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) for details.

---