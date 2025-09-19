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

## DeepEval: Evaluate and Improve Your LLMs with Ease

DeepEval is an open-source framework designed to simplify the evaluation and testing of Large Language Model (LLM) applications. Built with the latest research, DeepEval empowers developers to rigorously test LLM outputs using a variety of metrics. [Explore DeepEval on GitHub](https://github.com/confident-ai/deepeval).

**Key Features:**

*   **Comprehensive Metrics:** Evaluate your LLMs with a rich set of metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG Metrics: Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS.
    *   Agentic Metrics: Task Completion, Tool Correctness.
    *   Other Metrics: Hallucination, Summarization, Bias, Toxicity.
    *   Conversational Metrics: Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence.
*   **Custom Metric Creation:** Easily build and integrate your own custom evaluation metrics.
*   **Synthetic Dataset Generation:** Generate synthetic datasets to cover a wide range of test cases.
*   **CI/CD Integration:** Seamlessly integrate DeepEval into any CI/CD environment.
*   **Red Teaming Capabilities:** Identify over 40 safety vulnerabilities in your LLM applications, including toxicity, bias, and prompt injection, with advanced attack enhancement strategies.
*   **LLM Benchmark:** Effortlessly benchmark your LLMs on popular LLM benchmarks, such as MMLU, HellaSwag, and more.
*   **Confident AI Integration:** 100% integrated with Confident AI to provide the full evaluation lifecycle:
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection
*   **Component-Level Evaluation:** DeepEval supports both end-to-end and component-level evaluation, allowing you to pinpoint areas for improvement within your LLM systems.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

## Integrations

*   🦄 **LlamaIndex:** Unit test your RAG applications within your CI/CD pipelines.
*   🤗 **Hugging Face:** Enable real-time evaluations during LLM fine-tuning.

## Quickstart

Get started with DeepEval in just a few steps!

### Installation

```bash
pip install -U deepeval
```

### Setting Up an Account (Recommended)

Create a DeepEval account on the [Confident AI platform](https://confident-ai.com?utm_source=GitHub) to gain access to advanced features and cloud-based reporting.

```bash
deepeval login
```

Follow the instructions in the CLI to authenticate and save your API key. All test results are then automatically tracked (learn more about data privacy [here](https://deepeval.com/docs/data-privacy?utm_source=GitHub)).

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Write an end-to-end evaluation using `test_chatbot.py`:

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

Set your `OPENAI_API_KEY` environment variable.  You can also evaluate using a custom model, as detailed in [our docs](https://deepeval.com/docs/metrics-introduction#using-a-custom-llm?utm_source=GitHub).

```bash
export OPENAI_API_KEY="..."
```

Run your test:

```bash
deepeval test run test_chatbot.py
```

**Congratulations!** The test should pass ✅.

-   `input`: User input.
-   `actual_output`: Placeholder for your application's output.
-   `expected_output`: The ideal response to a given input.
-   `GEval`: A research-backed metric for LLM output evaluation.
-   The metric `criteria` assesses the correctness of the `actual_output` against the `expected_output`.
-   Metric scores range from 0-1; the `threshold` determines if the test passes.

For additional evaluation options, metrics, and tutorials, check out the [documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub).

### Evaluating Nested Components

Use component-level evaluations for detailed insights. The `@observe` decorator helps apply metrics at this level:

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

Alternatively, use the `evaluate` function directly:

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

DeepEval's modularity allows standalone metric usage:

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

### Evaluating Datasets in Bulk

Here's how to evaluate a dataset:

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

Run in the CLI:

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

Or, without Pytest:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

### A Note on Env Variables (.env / .env.local)

DeepEval automatically loads `.env.local` then `.env` from your current working directory **at import time**.

**Precedence:** process env -> `.env.local` -> `.env`.

Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

## DeepEval with Confident AI

Enhance your evaluation process with [Confident AI](https://confident-ai.com?utm_source=Github), the DeepEval cloud platform:

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

Log in from the CLI:

```bash
deepeval login
```

Then rerun your tests, and view the results via the CLI-provided link.

![Demo GIF](assets/demo.gif)

## Configuration

### Environment Variables via .env Files

Using `.env.local` or `.env` is optional. DeepEval uses existing environment variables if these files are missing.

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br />

## Contributing

Refer to [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contribution guidelines.

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

Developed by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

<br />

## License

DeepEval is licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).