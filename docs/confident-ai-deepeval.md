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


## DeepEval: Evaluate and Test Your LLMs with Ease

DeepEval is an open-source LLM evaluation framework that helps you test and refine your large language model applications with advanced metrics, seamless integrations, and comprehensive testing capabilities; [visit the original repo](https://github.com/confident-ai/deepeval).

### Key Features:

*   **Comprehensive Metrics:** Evaluate your LLMs using a wide range of metrics, including G-Eval, RAG metrics (Answer Relevancy, Faithfulness, etc.), agentic metrics (Task Completion, Tool Correctness), and more – all running locally on your machine.
*   **Component-Level and End-to-End Evaluation:**  Test your entire LLM application or individual components like LLM calls, retrievers, and agents.
*   **Custom Metric Creation:** Easily build and integrate your own custom metrics to tailor your evaluations to your specific needs.
*   **Synthetic Dataset Generation:** Create synthetic datasets for robust LLM evaluation.
*   **CI/CD Integration:** Integrate DeepEval seamlessly into any CI/CD environment.
*   **Red Teaming Capabilities:** Identify and mitigate LLM vulnerabilities with built-in red-teaming features, including tests for toxicity, bias, and SQL injection.
*   **Benchmarking:** Quickly benchmark your LLMs on popular benchmarks like MMLU, HellaSwag, and HumanEval.
*   **Confident AI Integration:**  Leverage the full evaluation lifecycle with Confident AI platform for data curation, benchmarking, result comparison, debugging, and continuous monitoring.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br />

##  Metrics and Features: Detailed Overview

*   **Supports both end-to-end and component-level LLM evaluation.**
*   **Ready-to-use LLM Evaluation Metrics:**

    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG Metrics:**
        *   Answer Relevancy
        *   Faithfulness
        *   Contextual Recall
        *   Contextual Precision
        *   Contextual Relevancy
        *   RAGAS
    *   **Agentic Metrics:**
        *   Task Completion
        *   Tool Correctness
    *   **Other Metrics:**
        *   Hallucination
        *   Summarization
        *   Bias
        *   Toxicity
    *   **Conversational Metrics:**
        *   Knowledge Retention
        *   Conversation Completeness
        *   Conversation Relevancy
        *   Role Adherence
    *   **And more!**
*   **Custom Metrics:** Build and integrate your own metrics.
*   **Synthetic Dataset Generation:** Create datasets for evaluation.
*   **CI/CD Integration:** Works seamlessly with your CI/CD pipeline.
*   **Red Teaming:**  Test for over 40 safety vulnerabilities with advanced attack strategies.
*   **Benchmarking:** Easily benchmark LLMs on popular benchmarks.
*   **Full lifecycle with Confident AI:**  Curate datasets, compare iterations, debug results, and monitor performance on the [DeepEval platform](https://confident-ai.com?utm_source=GitHub).

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

## 🔌 Integrations

*   🦄 LlamaIndex:  [Unit test RAG applications in CI/CD](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 Hugging Face:  [Enable real-time evaluations during LLM fine-tuning](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br />

## 🚀 Quickstart: Get Started Quickly

This quickstart guide demonstrates how to use DeepEval to evaluate a RAG-based customer support chatbot.

### Installation

```bash
pip install -U deepeval
```

### Create an Account (Highly Recommended)

Using the `deepeval` platform allows you to generate sharable testing reports on the cloud for free.

To login, run:

```bash
deepeval login
```

Follow the instructions in the CLI to create an account and paste your API key.

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Populate `test_chatbot.py` with your first test case for end-to-end evaluation:

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

Run your test case:

```bash
deepeval test run test_chatbot.py
```

**Congratulations!**

*   The `input` is your user query, and `actual_output` is your application's response.
*   `expected_output` is the ideal answer.  `GEval` evaluates the correctness of your app's output.
*   Metric scores range from 0-1; the `threshold` determines pass/fail.

[Read the documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more on metrics, custom metrics, and integrations.

<br />

## Evaluating Nested Components

Use **component-level** evals to evaluate components within your LLM application.

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

Learn more about component-level evaluation [here.](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

<br />

##  Evaluating Without Pytest Integration

Evaluate without pytest in a notebook environment:

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

DeepEval provides modular metrics:

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

Refer to the docs to select appropriate metrics for your use case.

## Evaluating a Dataset/Test Cases in Bulk

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

Run in CLI:

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

Alternative (without Pytest):

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

## LLM Evaluation with Confident AI

The best LLM evaluation lifecycle is only achievable with [the DeepEval platform](https://confident-ai.com?utm_source=Github).

1.  Curate/annotate evaluation datasets.
2.  Benchmark and compare LLM apps.
3.  Fine-tune metrics.
4.  Debug results.
5.  Monitor and improve in-product performance.
6.  Iterate for optimal results.

To begin, login from the CLI:

```bash
deepeval login
```

Now, run your test file:

```bash
deepeval test run test_chatbot.py
```

View results in the browser!

![Demo GIF](assets/demo.gif)

<br />

## Contributing

Review the [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contribution guidelines.

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

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.