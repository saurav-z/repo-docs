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

<!-- SEO-Optimized Summary -->
<p align="center">
  <b>Revolutionize your LLM testing with DeepEval, the open-source framework for comprehensive LLM evaluation, designed to help you build and deploy high-quality, reliable AI applications.</b>
</p>

DeepEval is a powerful, open-source framework designed for evaluating and testing Large Language Model (LLM) systems.  Think of it as Pytest, but specifically tailored for unit testing LLM outputs, incorporating the latest research to provide robust evaluation metrics.  DeepEval empowers developers to assess and refine LLM applications efficiently.  [Explore the DeepEval repository on GitHub](https://github.com/confident-ai/deepeval).

<br />

## Key Features

*   **Comprehensive Evaluation:** Supports both end-to-end and component-level LLM evaluation, allowing for detailed testing across your entire application or within specific modules.
*   **Diverse Metrics:** Offers a wide range of ready-to-use LLM evaluation metrics powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG metrics:**
        *   Answer Relevancy
        *   Faithfulness
        *   Contextual Recall
        *   Contextual Precision
        *   Contextual Relevancy
        *   RAGAS
    *   **Agentic metrics:**
        *   Task Completion
        *   Tool Correctness
    *   **Others:**
        *   Hallucination
        *   Summarization
        *   Bias
        *   Toxicity
    *   **Conversational metrics:**
        *   Knowledge Retention
        *   Conversation Completeness
        *   Conversation Relevancy
        *   Role Adherence
    *   Custom Metric Creation: Build your own custom metrics that seamlessly integrate into DeepEval's ecosystem.
*   **Synthetic Dataset Generation:** Generates synthetic datasets to comprehensively evaluate the performance of your LLM application.
*   **CI/CD Integration:** Integrates seamlessly with **ANY** CI/CD environment for automated testing and continuous monitoring.
*   **Red Teaming:** Red team your LLM application for 40+ safety vulnerabilities in a few lines of code, including:
    *   Toxicity
    *   Bias
    *   SQL Injection
    *   etc., using advanced 10+ attack enhancement strategies such as prompt injections.
*   **Benchmark Support:** Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://deepeval.com/docs/benchmarks-introduction?utm_source=GitHub), which includes:
    *   MMLU
    *   HellaSwag
    *   DROP
    *   BIG-Bench Hard
    *   TruthfulQA
    *   HumanEval
    *   GSM8K
*   **Confident AI Integration:** [100% integrated with Confident AI](https://confident-ai.com?utm_source=GitHub) for the full evaluation lifecycle:
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

## Integrations

*   🦄 **LlamaIndex**:  [Unit test RAG applications in CI/CD](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub).
*   🤗 **Hugging Face**: [Enable real-time evaluations during LLM fine-tuning](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub).

<br />

## QuickStart

Easily evaluate your LLM applications with the DeepEval framework.

### Installation

```bash
pip install -U deepeval
```

### Account Creation (Highly Recommended)

Signing up for the `deepeval` platform allows you to generate shareable test reports on the cloud. It is free and takes no additional code to set up.

To login, run:

```bash
deepeval login
```

Follow the CLI instructions to create an account and enter your API key. All test cases will automatically be logged (find more information on data privacy [here](https://deepeval.com/docs/data-privacy?utm_source=GitHub)).

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Populate `test_chatbot.py` with the following code:

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

Finally, run the test:

```bash
deepeval test run test_chatbot.py
```

**Success!** Here's what happened:

*   `input`: Mimics user input.
*   `actual_output`: Placeholder for your application's output.
*   `expected_output`: The ideal answer.
*   `GEval`: A research-backed metric to evaluate your LLM output's accuracy.
*   The metric's `criteria` assesses the correctness of the `actual_output` against the `expected_output`.
*   Metric scores range from 0-1, and `threshold=0.5` determines pass/fail.

[Read the documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for advanced usage, metrics, custom metrics, and integrations.

<br />

## Evaluating Nested Components

Apply component-level evaluations by using the `@observe` decorator on specific components to apply metrics.

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

## Evaluating Without Pytest Integration

Evaluate in a notebook environment without Pytest:

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

DeepEval's modular design allows for easy use of individual metrics:

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

Refer to the documentation to choose the right metric for your use case.

## Evaluating a Dataset / Test Cases in Bulk

Evaluate a dataset of test cases in DeepEval:

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

Alternatively, evaluate a dataset without Pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

<br />

## LLM Evaluation With Confident AI

Unlock the full potential of the DeepEval framework with [the DeepEval platform](https://confident-ai.com?utm_source=Github).

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

Learn more about Confident AI [here](https://www.confident-ai.com/docs?utm_source=GitHub).

Start by logging in from the CLI:

```bash
deepeval login
```

Follow the instructions to log in and enter your API key.

Now, rerun your test file:

```bash
deepeval test run test_chatbot.py
```

A link will appear in the CLI. Open it in your browser to view the results!

![Demo GIF](assets/demo.gif)

<br />

## Contributing

Refer to [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on code of conduct and contribution guidelines.

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

Developed by the founders of Confident AI. For inquiries, contact jeffreyip@confident-ai.com.

<br />

## License

DeepEval is licensed under Apache 2.0; see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) for details.