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

## Evaluate and Improve Your LLM Applications with DeepEval

DeepEval is an open-source LLM evaluation framework that empowers you to rigorously test, measure, and refine your large language model applications. Check out the [DeepEval repository](https://github.com/confident-ai/deepeval) to get started!

**Key Features:**

*   **Comprehensive Metrics:** Evaluate LLM outputs using a wide range of metrics, including G-Eval, RAG metrics (Answer Relevancy, Faithfulness, etc.), agentic metrics, and more, all running locally on your machine.
*   **Flexible Evaluation:** Support for both end-to-end and component-level LLM evaluation, giving you granular control over your testing.
*   **Customizable:** Build and integrate your own custom metrics seamlessly within the DeepEval ecosystem.
*   **Synthetic Dataset Generation:** Create synthetic datasets for robust and comprehensive evaluation.
*   **CI/CD Integration:** Integrates smoothly with any CI/CD environment for automated testing.
*   **Red Teaming Capabilities:** Red team your LLM application in just a few lines of code, including toxicity, bias, SQL injection, and more!
*   **Benchmarking:** Easily benchmark your LLM on popular LLM benchmarks (MMLU, HellaSwag, etc.) in under 10 lines of code.
*   **Confident AI Integration:** 100% integrated with Confident AI for a complete evaluation lifecycle, including dataset curation, benchmarking, metric fine-tuning, debugging, and monitoring.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br />

## 🚀 Quickstart: Evaluate Your RAG-Based Chatbot

Here's how DeepEval can help you test a RAG-based customer support chatbot.

### 1. Installation

```bash
pip install -U deepeval
```

### 2. Account Creation (Highly Recommended)

Create a free account on the [DeepEval platform](https://confident-ai.com?utm_source=GitHub) to share testing reports, track progress, and collaborate effectively.

Login from the CLI:

```
deepeval login
```

Follow the instructions to create an account and securely store your API key.

### 3. Create a Test File

Create a test file:

```bash
touch test_chatbot.py
```

### 4. Write Your First Test Case

In `test_chatbot.py`, create an end-to-end evaluation:

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

### 5. Set Your API Key and Run Tests

Set your `OPENAI_API_KEY` environment variable and run the test:

```
export OPENAI_API_KEY="..."
deepeval test run test_chatbot.py
```

**Congratulations!** This test case will verify your chatbot’s output against expectations.

**Explanation:**

*   `input`: User query.
*   `actual_output`: Your LLM app's response.
*   `expected_output`: Ideal response.
*   `GEval`: A research-backed metric to evaluate output accuracy.
*   `threshold`: Determines pass/fail (0-1).

[Explore our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more advanced options.

<br />

## Component-Level Evaluations

Use the `@observe` decorator to evaluate specific components in your LLM app:

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

<br />

## Evaluating Without Pytest

For notebook environments:

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

<br />

## Standalone Metrics

Use individual metrics:

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
print(answer_relevancy_metric.reason)
```

<br />

## Bulk Dataset Evaluation

Evaluate a dataset:

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

Run in the CLI (with optional parallel testing):

```bash
deepeval test run test_<filename>.py -n 4
```

Alternatively, evaluate without Pytest:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

<br />

## Environment Variables

DeepEval automatically loads `.env.local` then `.env` at import.

**Precedence:** process env -> `.env.local` -> `.env`

Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br />

## DeepEval with Confident AI

[Confident AI](https://confident-ai.com?utm_source=Github) provides:

1.  Dataset curation/annotation
2.  LLM app benchmarking
3.  Metric fine-tuning
4.  Evaluation result debugging
5.  In-product response monitoring

Login from the CLI:

```bash
deepeval login
```

Then, re-run your test file, and view the results in your browser!

![Demo GIF](assets/demo.gif)

<br />

## Configuration

### Environment variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)

<br />

## Contributing

See [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contribution guidelines.

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

DeepEval is licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).