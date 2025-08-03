<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">DeepEval: The Ultimate LLM Evaluation Framework</h1>
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


## Optimize Your LLMs with DeepEval: The Open-Source LLM Evaluation Framework

DeepEval is a powerful, open-source framework designed to streamline the evaluation and testing of Large Language Models (LLMs).  Built to be like Pytest but specialized for LLM output evaluation, DeepEval utilizes the latest research to assess LLM performance using a variety of metrics, empowering you to build more robust and reliable LLM applications.  **<a href="https://github.com/confident-ai/deepeval">Check out the original repo!</a>**

### Key Features

*   **Comprehensive Metric Suite:**
    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG Metrics:** Answer Relevancy, Faithfulness, Contextual Recall & Precision, Contextual Relevancy, RAGAS
    *   **Agentic Metrics:** Task Completion, Tool Correctness
    *   **Other Metrics:** Hallucination, Summarization, Bias, Toxicity
    *   **Conversational Metrics:** Knowledge Retention, Conversation Completeness & Relevancy, Role Adherence
    *   And many more!
*   **Flexible Evaluation:** Supports both end-to-end and component-level LLM evaluations.
*   **Customization:** Build and integrate your own custom metrics seamlessly.
*   **Synthetic Dataset Generation:** Create datasets for robust testing.
*   **CI/CD Integration:** Works seamlessly with your existing CI/CD pipelines.
*   **Red Teaming:** Identify and mitigate over 40 safety vulnerabilities (Toxicity, Bias, SQL Injection, etc.) with advanced attack strategies.
*   **Benchmarking:** Easily benchmark your LLMs on popular benchmarks like MMLU, HellaSwag, and more (under 10 lines of code).
*   **Confident AI Integration:** Leverage the full LLM evaluation lifecycle with the [DeepEval platform](https://confident-ai.com?utm_source=GitHub):
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection

>   **[Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.**

<br />

# 🔌 Integrations

*   🦄 [LlamaIndex](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub): Unit test your RAG applications in CI/CD.
*   🤗 [Hugging Face](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub): Enable real-time evaluations during LLM fine-tuning.

<br />

# 🚀 QuickStart

## Installation

```bash
pip install -U deepeval
```

## Create an Account (Highly Recommended)

Using the `deepeval` platform will allow you to generate sharable testing reports on the cloud. It is free, takes no additional code to setup, and we highly recommend giving it a try.

To login, run:

```bash
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste it into the CLI. All test cases will automatically be logged (find more information on data privacy [here](https://deepeval.com/docs/data-privacy?utm_source=GitHub)).

## Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case to run an **end-to-end** evaluation using DeepEval:

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

Set your `OPENAI_API_KEY` as an environment variable (you can also evaluate using your own custom model, for more details visit [this part of our docs](https://deepeval.com/docs/metrics-introduction#using-a-custom-llm?utm_source=GitHub)):

```bash
export OPENAI_API_KEY="..."
```

And finally, run `test_chatbot.py` in the CLI:

```bash
deepeval test run test_chatbot.py
```

**Congratulations! Your test case should have passed ✅**

## Evaluating Nested Components (Component-Level Evaluation)

Trace components within your LLM application using the `@observe` decorator:

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

Learn everything about component-level evaluations [here.](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

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

## Evaluating a Dataset / Test Cases in Bulk

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

<br/>

Alternatively, although we recommend using `deepeval test run`, you can evaluate a dataset/test cases without using our Pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

# LLM Evaluation With Confident AI

The correct LLM evaluation lifecycle is only achievable with [the DeepEval platform](https://confident-ai.com?utm_source=Github). It allows you to:

1. Curate/annotate evaluation datasets on the cloud
2. Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3. Fine-tune metrics for custom results
4. Debug evaluation results via LLM traces
5. Monitor & evaluate LLM responses in product to improve datasets with real-world data
6. Repeat until perfection

Everything on Confident AI, including how to use Confident is available [here](https://documentation.confident-ai.com/docs?utm_source=GitHub).

To begin, login from the CLI:

```bash
deepeval login
```

Follow the instructions to log in, create your account, and paste your API key into the CLI.

Now, run your test file again:

```bash
deepeval test run test_chatbot.py
```

You should see a link displayed in the CLI once the test has finished running. Paste it into your browser to view the results!

![Demo GIF](assets/demo.gif)

<br />

# Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

<br />

# Roadmap

Features:

-   [x] Integration with Confident AI
-   [x] Implement G-Eval
-   [x] Implement RAG metrics
-   [x] Implement Conversational metrics
-   [x] Evaluation Dataset Creation
-   [x] Red-Teaming
-   [ ] DAG custom metrics
-   [ ] Guardrails

<br />

# Authors

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all enquiries.

<br />

# License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.