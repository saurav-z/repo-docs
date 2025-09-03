<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval">
        <img alt="GitHub stars" src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social">
    </a>
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/3SEyvpgu2f?style=flat">
    </a>
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
    <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
    <a href="#metrics-and-features">Metrics and Features</a> |
    <a href="#quickstart">Quickstart</a> |
    <a href="#integrations">Integrations</a> |
    <a href="https://confident-ai.com?utm_source=GitHub">DeepEval Platform</a>
</p>

DeepEval simplifies LLM evaluation, offering a powerful, open-source framework for testing and refining your large language model applications.

## Key Features:

*   **Comprehensive Metrics:** Evaluate LLMs with a wide range of ready-to-use metrics:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG Metrics: Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS
    *   Agentic Metrics: Task Completion, Tool Correctness
    *   Other Metrics: Hallucination, Summarization, Bias, Toxicity, etc.
    *   Conversational Metrics: Knowledge Retention, Conversation Completeness/Relevancy, Role Adherence
*   **Custom Metric Creation:** Build and integrate your own custom evaluation metrics seamlessly.
*   **Synthetic Dataset Generation:** Create datasets for comprehensive LLM evaluation.
*   **CI/CD Integration:** Integrates with any CI/CD environment.
*   **Red Teaming:** Identify safety vulnerabilities (Toxicity, Bias, SQL Injection, etc.) in your LLM applications.
*   **LLM Benchmarking:** Benchmark LLMs on popular benchmarks (MMLU, HellaSwag, etc.) with minimal code.
*   **Confident AI Integration:** Full lifecycle evaluation support with [Confident AI](https://confident-ai.com?utm_source=GitHub):
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

## Quickstart

Get started evaluating your LLM applications in minutes.

### Installation

```bash
pip install -U deepeval
```

### Login (Highly Recommended)

Leverage the power of the DeepEval platform for sharable reports.

```bash
deepeval login
```

### Example Test Case (End-to-End Evaluation)

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

Set your `OPENAI_API_KEY` environment variable and run:

```bash
export OPENAI_API_KEY="..."
deepeval test run test_chatbot.py
```

**Learn more:**  [DeepEval Documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub).

### Evaluating Nested Components (Component-Level Evaluation)

Use the `@observe` decorator to trace and evaluate components within your LLM application.

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
**Learn more:** [Component Level Evaluation Docs](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

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

## Integrations

*   🦄 LlamaIndex: Unit test RAG applications in CI/CD.
*   🤗 Hugging Face: Enable real-time evaluations during LLM fine-tuning.

## LLM Evaluation With Confident AI

Achieve the complete LLM evaluation lifecycle with [the DeepEval platform](https://confident-ai.com?utm_source=Github).

### Steps:

1.  Curate/annotate datasets.
2.  Benchmark and compare models/prompts.
3.  Fine-tune metrics.
4.  Debug results.
5.  Monitor and improve with real-world data.

To begin, login from the CLI:

```bash
deepeval login
```

Now, run your test file again:

```bash
deepeval test run test_chatbot.py
```

View your results via the link provided in the CLI.

## Contributing

See [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on contributing.

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

DeepEval is licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).

[Back to Top](#deepeval-the-open-source-llm-evaluation-framework)