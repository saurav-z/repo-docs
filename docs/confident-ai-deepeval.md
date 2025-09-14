<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: Your Open-Source LLM Evaluation Framework</h1>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social">
    </a>
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="Discord" src="https://img.shields.io/discord/1188578870372082768?logo=discord&style=social">
    </a>
    <a href="https://twitter.com/deepeval">
        <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/deepeval?style=social&logo=x">
    </a>
</p>

<p align="center">
  <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
  <a href="#-key-features">Key Features</a> |
  <a href="#-quickstart">Quickstart</a> |
  <a href="#-integrations">Integrations</a> |
  <a href="https://confident-ai.com?utm_source=GitHub">DeepEval Platform</a>
</p>

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
</p>

DeepEval is an open-source framework that simplifies evaluating and testing your LLM applications, offering a robust solution for ensuring quality and reliability. 

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

## ✨ Key Features

*   **Comprehensive Metrics:** Evaluate LLM outputs with a wide array of ready-to-use metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG Metrics (Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS)
    *   Agentic Metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, Toxicity, and more.
    *   Conversational Metrics (Knowledge Retention, Completeness, Relevancy, Role Adherence)
*   **Customization:** Easily build and integrate your own custom metrics tailored to your specific needs.
*   **Local Execution:** All evaluations are performed locally on your machine, ensuring data privacy and control.
*   **CI/CD Integration:** Seamlessly integrates with any CI/CD environment for automated testing.
*   **Red Teaming:** Identify and mitigate safety vulnerabilities with built-in red-teaming capabilities.
*   **Benchmarking:** Quickly benchmark your LLMs on popular benchmarks.
*   **Confident AI Integration:** Full integration with the DeepEval platform for comprehensive evaluation lifecycle management, including:
    *   Dataset curation and annotation.
    *   Benchmarking and comparison.
    *   Metric fine-tuning.
    *   LLM trace debugging.
    *   Real-world data monitoring and dataset improvement.

## 🔌 Integrations

*   🦄 **LlamaIndex:** Unit test your RAG applications within your CI/CD pipeline.
*   🤗 **Hugging Face:** Enable real-time evaluations during LLM fine-tuning.

## 🚀 Quickstart

Get started with DeepEval in minutes!  For the full evaluation lifecycle, [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub).

### 1. Installation

```bash
pip install -U deepeval
```

### 2.  Create an account (recommended)

Run the following in your terminal:

```bash
deepeval login
```

Follow the CLI instructions to create your account, copy your API key, and then paste it back into your CLI.

### 3. Writing Your First Test Case

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

### 4. Set up your API Key

```bash
export OPENAI_API_KEY="..." # Replace with your actual OpenAI API key
```

### 5. Run the Test

```bash
deepeval test run test_chatbot.py
```

### 6. What Happened?

*   `input`:  User input.
*   `actual_output`: The output from your LLM application.
*   `expected_output`: The ideal response.
*   `GEval`:  A metric provided by `deepeval` to evaluate your LLM output with human-like accuracy.
*   `criteria`: The correctness of `actual_output` against `expected_output`.
*   Scores range from 0-1, with `threshold=0.5` determining pass/fail.

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) to learn more.

### Evaluating Nested Components

Component-level evaluations allow you to evaluate specific components within your LLM application. Trace components (LLM calls, retrievers, tools) using the `@observe` decorator:

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

Learn more about component-level evaluations [here.](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

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

### Standalone Metrics

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

### Evaluating Datasets

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

### Alternative Dataset Evaluation (without Pytest)

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

## LLM Evaluation With Confident AI

Unlock the full potential of DeepEval with the [DeepEval platform](https://confident-ai.com?utm_source=Github) for a complete evaluation lifecycle:

1.  Curate and annotate datasets.
2.  Benchmark and compare LLM applications.
3.  Fine-tune metrics.
4.  Debug evaluation results.
5.  Monitor and improve datasets with real-world data.

**Log in from the CLI:**

```bash
deepeval login
```

Follow the instructions.

Now, run your test file again:

```bash
deepeval test run test_chatbot.py
```

Find the results on the DeepEval platform by clicking on the link in the CLI.

![Demo GIF](assets/demo.gif)

## Configuration

### Environment Variables

Use `.env.local` or `.env` for configuration. DeepEval uses existing environment variables if the files are missing.

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

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

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.