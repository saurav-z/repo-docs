<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">DeepEval: The LLM Evaluation Framework</h1>
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

## DeepEval: Supercharge Your LLM Development with Comprehensive Evaluation

DeepEval is an open-source framework, similar to Pytest, that empowers you to thoroughly test and evaluate your large language model (LLM) applications.  Visit the [DeepEval GitHub repository](https://github.com/confident-ai/deepeval) to get started!

**Key Features of DeepEval:**

*   ✅ **Comprehensive Metrics:** Evaluate LLM outputs using a diverse range of metrics:
    *   G-Eval
    *   DAG
    *   RAG metrics (Answer Relevancy, Faithfulness, Contextual Recall, Precision, and Relevancy, RAGAS)
    *   Agentic metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, Toxicity
    *   Conversational Metrics (Knowledge Retention, Conversation Completeness, Relevancy, Role Adherence)
    *   And more!
*   ⚙️ **Customizable:** Build and integrate your own custom metrics seamlessly.
*   🧪 **Synthetic Data Generation:** Generate synthetic datasets for robust evaluation.
*   🚀 **CI/CD Integration:**  Integrates effortlessly with any CI/CD environment.
*   🛡️ **Red Teaming:**  Test for 40+ safety vulnerabilities (toxicity, bias, prompt injection, etc.) with advanced attack strategies.
*   📊 **Benchmarking:**  Easily benchmark LLMs on popular benchmarks like MMLU, HellaSwag, and more.
*   ☁️ **Confident AI Integration:**  Leverage the DeepEval platform for:
    *   Dataset curation and annotation.
    *   LLM app benchmarking and comparison.
    *   Metric fine-tuning.
    *   Evaluation result debugging.
    *   Real-world data monitoring and improvement.

> [!IMPORTANT]
> Enhance your DeepEval testing by signing up for the free [DeepEval platform](https://confident-ai.com?utm_source=GitHub).  Compare iterations, generate shareable reports, and more!

> 📢 Join our Discord community for LLM evaluation discussions and support: [Discord](https://discord.com/invite/3SEyvpgu2f).

<br />

## 🔌 Integrations

*   🦄 **LlamaIndex:** Unit test RAG applications in CI/CD ([LlamaIndex Integration](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)).
*   🤗 **Hugging Face:** Enable real-time evaluations during LLM fine-tuning ([Hugging Face Integration](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)).

<br />

## 🚀 Quickstart

Get started with DeepEval in these simple steps.

### 1. Installation

```bash
pip install -U deepeval
```

### 2. Create an account (highly recommended)

The `deepeval` platform enables sharing and cloud-based testing reports. It's free and takes little setup.

```bash
deepeval login
```

Follow the CLI instructions to create or log in to your account.

### 3. Write Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case to run an **end-to-end** evaluation using DeepEval, which treats your LLM app as a black-box:

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

```
export OPENAI_API_KEY="..."
```

And finally, run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

**Congratulations! Your test case should have passed ✅** Let's breakdown what happened.

- The variable `input` mimics a user input, and `actual_output` is a placeholder for what your application's supposed to output based on this input.
- The variable `expected_output` represents the ideal answer for a given `input`, and [`GEval`](https://deepeval.com/docs/metrics-llm-evals) is a research-backed metric provided by `deepeval` for you to evaluate your LLM output's on any custom with human-like accuracy.
- In this example, the metric `criteria` is correctness of the `actual_output` based on the provided `expected_output`.
- All metric scores range from 0 - 1, which the `threshold=0.5` threshold ultimately determines if your test have passed or not.

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more information on more options to run end-to-end evaluation, how to use additional metrics, create your own custom metrics, and tutorials on how to integrate with other tools like LangChain and LlamaIndex.

<br />

### 4. Evaluating Nested Components

For component-level evaluations, use the `@observe` decorator.  This allows you to apply metrics to LLM calls, retrievers, tool calls, etc. (See [component-level evaluations](https://www.deepeval.com/docs/evaluation-component-level-llm-evals) for more details).

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

<br />

### 5. Evaluating Without Pytest Integration

Evaluate without pytest:

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

### 6. Using Standalone Metrics

DeepEval's modular design allows for easy metric usage:

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

### 7. Evaluating a Dataset / Test Cases in Bulk

Evaluate multiple test cases using datasets:

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

## LLM Evaluation With Confident AI

Leverage the complete LLM evaluation lifecycle with [the DeepEval platform](https://confident-ai.com?utm_source=Github):

1.  Curate/annotate evaluation datasets on the cloud.
2.  Benchmark LLM apps and compare iterations.
3.  Fine-tune metrics.
4.  Debug evaluation results.
5.  Monitor and evaluate responses in production.
6.  Iterate for continuous improvement.

Access comprehensive documentation [here](https://documentation.confident-ai.com/docs?utm_source=GitHub).

Login via the CLI:

```bash
deepeval login
```

Follow the instructions to log in and enter your API key.

Run your test file:

```bash
deepeval test run test_chatbot.py
```

View your results by opening the link provided in the CLI.

![Demo GIF](assets/demo.gif)

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

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.