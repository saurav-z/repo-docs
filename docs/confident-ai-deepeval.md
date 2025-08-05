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

## DeepEval: Evaluate and Improve Your LLM Applications with Ease

DeepEval is an open-source evaluation framework, akin to Pytest but built specifically for unit testing and evaluating Large Language Model (LLM) applications, providing the tools you need to test, benchmark, and improve your LLM-powered systems.  Explore the [DeepEval GitHub repository](https://github.com/confident-ai/deepeval) to get started!

**Key Features:**

*   **Comprehensive Metrics:**  Evaluate LLM outputs with a variety of metrics, including:
    *   G-Eval
    *   DAG
    *   RAG metrics (Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS)
    *   Agentic metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, Toxicity, and more.
    *   Conversational metrics (Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence)
*   **Custom Metric Support:**  Build and seamlessly integrate your own custom metrics.
*   **Synthetic Dataset Generation:** Generate synthetic datasets to test LLMs.
*   **CI/CD Integration:** Easily integrate with any CI/CD environment.
*   **Red Teaming:**  Test for over 40 safety vulnerabilities in LLM applications, including toxicity, bias, and prompt injections.
*   **LLM Benchmarking:**  Benchmark LLMs on popular datasets (MMLU, HellaSwag, DROP, etc.) with minimal code.
*   **Confident AI Integration:** Full integration with the DeepEval Platform for a complete evaluation lifecycle:  dataset curation, benchmarking, metric fine-tuning, debugging, monitoring, and improvement.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

## 🔌 Integrations

*   🦄 **LlamaIndex:** Unit test RAG applications in CI/CD.
*   🤗 **Hugging Face:** Enable real-time evaluations during LLM fine-tuning.

## 🚀 QuickStart

### Installation

```bash
pip install -U deepeval
```

### Create an Account (Recommended)

Logging into the `deepeval` platform allows you to generate sharable testing reports on the cloud.  It's free, requires no extra code, and is highly recommended.

```bash
deepeval login
```

Follow the CLI instructions to create an account and paste your API key.

### Writing Your First Test Case

Create a test file (e.g., `test_chatbot.py`) and add the following:

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

Set your `OPENAI_API_KEY` as an environment variable:

```bash
export OPENAI_API_KEY="..."
```

Run the test:

```bash
deepeval test run test_chatbot.py
```

**Congratulations! Your test case should have passed ✅**

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more information.

### Evaluating Nested Components

Use the `@observe` decorator for **component-level** evaluations:

```python
from deepeval.tracing import observe, update_current_span
from deepeval.test_case import LLMTestCase
from deepeval.dataset import Golden
from deepeval.metrics import GEval
from deepeval import evaluate

correctness = GEval(name="Correctness", criteria="Determine if the 'actual output' is correct based on the 'expected output'.", evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT])

@observe(metrics=[correctness])
def inner_component():
    # Component can be anything from an LLM call, retrieval, agent, tool use, etc.
    update_current_span(test_case=LLMTestCase(input="...", actual_output="..."))
    return

@observe
def llm_app(input: str):
    inner_component()
    return

evaluate(observed_callback=llm_app, goldens=[Golden(input="Hi!")])
```

### Evaluating Without Pytest

```python
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output="We offer a 30-day full refund at no extra costs.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
)
evaluate([test_case], [answer_relevancy_metric])
```

### Using Standalone Metrics

```python
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.7)
test_case = LLMTestCase(
    input="What if these shoes don't fit?",
    # Replace this with the actual output from your LLM application
    actual_output="We offer a 30-day full refund at no extra costs.",
    retrieval_context=["All customers are eligible for a 30 day full refund at no extra costs."]
)

answer_relevancy_metric.measure(test_case)
print(answer_relevancy_metric.score)
# All metrics also offer an explanation
print(answer_relevancy_metric.reason)
```

### Evaluating a Dataset / Test Cases in Bulk

```python
import pytest
from deepeval import assert_test
from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

dataset = EvaluationDataset(goldens=[Golden(input="What's the weather like today?")])

for golden in dataset.goldens:
    test_case = LLMTestCase(
        input=golden.input,
        actual_output=your_llm_app(golden.input)
    )
    dataset.add_test_case(test_case)

@pytest.mark.parametrize(
    "test_case",
    dataset,
)
def test_customer_chatbot(test_case: LLMTestCase):
    answer_relevancy_metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [answer_relevancy_metric])
```

Run with:
```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

Alternatively:
```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

## LLM Evaluation With Confident AI

Leverage the full evaluation lifecycle with [the DeepEval platform](https://confident-ai.com?utm_source=Github):

1.  Curate/annotate evaluation datasets.
2.  Benchmark LLM apps and compare iterations.
3.  Fine-tune metrics.
4.  Debug evaluation results.
5.  Monitor and improve LLM responses with real-world data.

To begin, login from the CLI:

```bash
deepeval login
```

Run your test file:

```bash
deepeval test run test_chatbot.py
```

View your results via the link displayed in the CLI.

![Demo GIF](assets/demo.gif)

## Contributing

See [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contribution guidelines.

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

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all inquiries.

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.