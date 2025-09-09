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

## 🚀 DeepEval: Effortlessly Evaluate and Test Your LLMs

DeepEval is an open-source LLM evaluation framework, enabling you to easily assess and improve your large language model applications. 

*   **[Check out the original repo for more details.](https://github.com/confident-ai/deepeval)**

## ✨ Key Features

*   **Comprehensive Evaluation:** Supports both end-to-end and component-level evaluation of LLMs.
*   **Rich Metric Suite:** Offers a wide range of ready-to-use metrics, including:
    *   G-Eval
    *   DAG (Directed Acyclic Graph)
    *   RAG Metrics (Answer Relevancy, Faithfulness, Contextual Recall & Precision, Contextual Relevancy, RAGAS)
    *   Agentic Metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, and Toxicity detection
    *   Conversational Metrics (Knowledge Retention, Conversation Completeness & Relevancy, Role Adherence)
*   **Customizable Metrics:** Build and integrate your own custom metrics seamlessly.
*   **Synthetic Dataset Generation:** Create datasets for thorough evaluation.
*   **CI/CD Integration:** Integrates smoothly with any CI/CD environment.
*   **Red Teaming Capabilities:** Red team your LLM applications with advanced safety checks for toxicity, bias, and more, via prompt injection.
*   **Benchmarking Made Easy:** Benchmark LLMs on popular datasets like MMLU, HellaSwag, and others in just a few lines of code.
*   **Confident AI Integration:** 100% integrated with the [DeepEval platform](https://confident-ai.com?utm_source=GitHub) for the complete LLM evaluation lifecycle:
    *   Curate and annotate evaluation datasets.
    *   Benchmark and compare LLM app iterations.
    *   Fine-tune metrics.
    *   Debug results using LLM traces.
    *   Monitor and evaluate LLM responses.
    *   Iterate and refine until perfection.

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

## 🔌 Integrations

*   **LlamaIndex:** Unit test RAG applications within CI/CD.
*   **Hugging Face:** Enable real-time evaluations during LLM fine-tuning.

## 🚀 Quickstart Guide

### Installation

```bash
pip install -U deepeval
```

### Environment Variables

DeepEval automatically loads `.env.local` and then `.env` files.  Set `DEEPEVAL_DISABLE_DOTENV=1` to opt out.

### Writing Your First Test Case

Create a test file, e.g., `test_chatbot.py`:

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

### Set your OPENAI_API_KEY
```
export OPENAI_API_KEY="..."
```

### Run the Test

```bash
deepeval test run test_chatbot.py
```

**Congratulations!** The test case should have passed!

## Component-Level Evaluation

Use the `@observe` decorator to evaluate individual components within your LLM applications.

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

## Evaluating Without Pytest

Evaluate directly in a notebook environment:

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

## Using Standalone Metrics

Use individual metrics as needed.

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

## Evaluating a Dataset

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

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

## LLM Evaluation with Confident AI

Leverage the full potential of the DeepEval platform for:

1.  Dataset curation and annotation.
2.  Benchmarking and comparison of LLM app iterations.
3.  Fine-tuning metrics.
4.  Debugging with LLM traces.
5.  Monitoring and evaluation of responses.
6.  Continuous improvement.

Log in from the CLI to get started:

```bash
deepeval login
```

## Configuration

### Environment Variables

Use `.env.local` or `.env` to manage environment variables.

**Precedence:** process env -> `.env.local` -> `.env`

### Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct and contribution process.

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

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.