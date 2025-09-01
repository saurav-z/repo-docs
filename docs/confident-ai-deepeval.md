<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>

<p align="center">
  <a href="https://github.com/confident-ai/deepeval">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social">
  </a>
  <a href="https://discord.gg/3SEyvpgu2f">
    <img alt="Discord" src="https://img.shields.io/discord/1163868819201636402?label=Discord&logo=discord&style=social">
  </a>
</p>

<p align="center">
  <a href="https://www.deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
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

**DeepEval is your go-to open-source solution for comprehensive LLM evaluation and testing.**  This powerful framework, similar to Pytest but designed specifically for LLMs, lets you rigorously assess your large language model applications.  [Explore DeepEval on GitHub](https://github.com/confident-ai/deepeval)

> [!IMPORTANT]
> Maximize your LLM app's potential with the DeepEval platform! Sign up [here](https://confident-ai.com?utm_source=GitHub) to compare iterations, generate reports, and more.

<br/>

## 🔑 Key Features

*   **End-to-End and Component-Level Evaluation:** Assess your LLMs at every stage.
*   **Rich Metric Library:**  Evaluate with a diverse set of metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG metrics (Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS)
    *   Agentic metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, Toxicity
    *   Conversational metrics (Knowledge Retention, Conversation Completeness/Relevancy, Role Adherence)
*   **Custom Metric Creation:** Easily build and integrate your own custom metrics.
*   **Synthetic Dataset Generation:** Create datasets for thorough evaluation.
*   **CI/CD Integration:**  Seamlessly integrates with your CI/CD pipelines.
*   **Red Teaming:**  Identify and mitigate over 40 safety vulnerabilities.
*   **LLM Benchmark Support:**  Benchmark your LLMs on popular datasets like MMLU, HellaSwag, DROP, and more.
*   **100% Integrated with Confident AI:** Leverage the complete evaluation lifecycle, including data curation, benchmarking, metric fine-tuning, debugging, monitoring, and iteration.

<br/>

## 🔌 Integrations

*   🦄 **LlamaIndex:**  [Unit test your RAG applications in CI/CD.](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 **Hugging Face:**  [Enable real-time evaluations during LLM fine-tuning.](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br/>

## 🚀 Quickstart

Get started testing your LLM applications with DeepEval in minutes!  This example shows how to test a customer support chatbot.

### Installation

```bash
pip install -U deepeval
```

### Account Creation (Highly Recommended)

Create a free account on the DeepEval platform to share test results.

```bash
deepeval login
```

Follow the instructions to create an account and save your API key.

### Create a Test File

Create a file, such as `test_chatbot.py`, and add the following test case:

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

Run your test:

```bash
deepeval test run test_chatbot.py
```

**Success!**  Learn more about customizing your tests, using additional metrics, and integrating with LangChain and LlamaIndex in the [DeepEval documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub).

<br/>

### Evaluating Nested Components

Component-level evaluations can be done using the `@observe` decorator.

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

<br/>

### Evaluating Without Pytest Integration

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

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

<br/>

Without Pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

<br/>

## LLM Evaluation with Confident AI

The DeepEval platform offers a complete LLM evaluation lifecycle, including:

1.  Dataset curation/annotation on the cloud
2.  Benchmarking and comparison
3.  Metric fine-tuning
4.  Debugging with LLM traces
5.  Monitoring in-product and iteration

To get started, log in to the CLI:

```bash
deepeval login
```

Run your test file:

```bash
deepeval test run test_chatbot.py
```

View your results!

![Demo GIF](assets/demo.gif)

<br/>

## Contributing

Contributions are welcome! Read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details.

<br/>

## Roadmap

*   [x] Integration with Confident AI
*   [x] Implement G-Eval
*   [x] Implement RAG metrics
*   [x] Implement Conversational metrics
*   [x] Evaluation Dataset Creation
*   [x] Red-Teaming
*   [ ] DAG custom metrics
*   [ ] Guardrails

<br/>

## Authors

Built by the Confident AI team.  Contact jeffreyip@confident-ai.com for inquiries.

<br/>

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file.