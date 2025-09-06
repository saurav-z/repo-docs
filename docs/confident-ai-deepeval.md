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

## DeepEval: Effortlessly Evaluate and Test Your LLMs

DeepEval is an open-source framework designed to simplify the evaluation and testing of your Large Language Model (LLM) applications, offering a robust, customizable, and efficient approach to ensure quality and reliability.  Dive deeper into the code at the [DeepEval GitHub repository](https://github.com/confident-ai/deepeval).

**Key Features:**

*   **Comprehensive Metrics:** Evaluate LLM outputs with a wide range of built-in metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG metrics: Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, and RAGAS
    *   Agentic metrics: Task Completion, Tool Correctness
    *   Others: Hallucination, Summarization, Bias, Toxicity
    *   Conversational metrics: Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence
*   **Custom Metric Creation:** Easily build and integrate your own custom evaluation metrics.
*   **Synthetic Dataset Generation:** Generate synthetic datasets for comprehensive evaluation.
*   **CI/CD Integration:** Seamlessly integrates with any CI/CD environment for automated testing.
*   **Red Teaming:**  Identify and mitigate over 40 safety vulnerabilities, including toxicity, bias, and prompt injections.
*   **LLM Benchmark Support:** Benchmark LLMs on popular benchmarks (MMLU, HellaSwag, DROP, etc.) in under 10 lines of code.
*   **Confident AI Integration:** Fully integrated with the DeepEval Platform, offering a complete evaluation lifecycle.  This includes:
    *   Dataset curation and annotation.
    *   LLM app benchmarking and comparison.
    *   Metric fine-tuning.
    *   Evaluation result debugging with LLM traces.
    *   Real-world data monitoring and dataset improvement.

> [!IMPORTANT]
>  Elevate your LLM evaluation with the [DeepEval platform](https://confident-ai.com?utm_source=GitHub), where you can share results, generate reports, and more.

> Want to discuss LLM evaluation or need help selecting the right metrics? [Join our Discord community](https://discord.com/invite/3SEyvpgu2f)!

<br />

## 🔌 Integrations

*   🦄 LlamaIndex:  [Unit test RAG applications in CI/CD](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 Hugging Face:  [Enable real-time evaluations during LLM fine-tuning](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br />

## 🚀 QuickStart

Get started by installing DeepEval:

```bash
pip install -U deepeval
```

### 1. Configure Environment Variables
Configure your environment variables using `.env` or `.env.local` files, or set them directly in your shell:

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

**Precedence:** process env -> `.env.local` -> `.env`

### 2. Create an Account (Recommended)
Creating a DeepEval account allows for sharable testing reports.

To log in:

```bash
deepeval login
```

Follow the CLI instructions to create an account and enter your API key.

### 3. Write a Test Case

Create a test file (e.g., `test_chatbot.py`):

```bash
touch test_chatbot.py
```

Add your first test case for end-to-end evaluation:

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

```
export OPENAI_API_KEY="..."
```

Run your test case:

```
deepeval test run test_chatbot.py
```

>  This should result in a successful test case!

*   `input`: Mimics user input.
*   `actual_output`: The output from your LLM application.
*   `expected_output`: The ideal answer.
*   `GEval`: A research-backed metric from DeepEval to assess LLM output correctness.
*   `criteria`: The basis for metric evaluation.
*   Metrics generate a score between 0-1.  `threshold=0.5` determines test success.

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for further customization, more metrics, custom metrics, and integration tutorials (LangChain, LlamaIndex, etc.).

<br />

## Component-Level Evaluation

Component-level evaluations allow you to test components within your LLM app. Use the `@observe` decorator to trace components like LLM calls, retrievers, tool calls, and agents, and apply metrics at a component level.

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
[Learn more about component-level evaluations](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

<br />

## Evaluating Without Pytest Integration

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

DeepEval's modular design allows standalone metric usage.

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
Make sure to use the docs to select the appropriate metrics for your project.

## Bulk Dataset Evaluation

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

Run the test cases:

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

Alternatively, you can evaluate datasets without our Pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

<br />

## LLM Evaluation with DeepEval Platform

To get the most out of DeepEval, leverage the [DeepEval platform](https://confident-ai.com?utm_source=Github) for the full evaluation lifecycle:

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using datasets, and compare iterations for optimal model/prompt performance.
3.  Fine-tune metrics.
4.  Debug evaluation results with LLM traces.
5.  Monitor and evaluate LLM responses in production to improve datasets with real-world data.
6.  Iterate for perfection.

To begin, log in from the CLI:

```bash
deepeval login
```

Then rerun your test file:

```bash
deepeval test run test_chatbot.py
```

Click the link provided in the CLI to see the results.

![Demo GIF](assets/demo.gif)

<br />

## Configuration

### Environment Variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br />

## Contributing

See [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contributing guidelines.

<br />

## Roadmap

*   [x] Integration with Confident AI
*   [x] Implement G-Eval
*   [x] Implement RAG metrics
*   [x] Implement Conversational metrics
*   [x] Evaluation Dataset Creation
*   [x] Red-Teaming
*   [ ] DAG custom metrics
*   [ ] Guardrails

<br />

## Authors

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

<br />

## License

DeepEval is licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) for details.