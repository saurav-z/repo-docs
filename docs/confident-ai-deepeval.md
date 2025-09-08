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

## DeepEval: Effortlessly Evaluate and Improve Your LLM Applications

DeepEval is an open-source LLM evaluation framework, similar to Pytest, designed to help you rigorously test and improve your large language model applications.  Visit the [DeepEval repository](https://github.com/confident-ai/deepeval) to get started.

**Key Features:**

*   **Comprehensive Metrics:** Evaluate LLM outputs with a wide range of ready-to-use metrics, including:
    *   G-Eval
    *   DAG
    *   RAG metrics (Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS)
    *   Agentic metrics (Task Completion, Tool Correctness)
    *   Hallucination, Summarization, Bias, Toxicity
    *   Conversational metrics (Knowledge Retention, Conversation Completeness/Relevancy, Role Adherence)
*   **Local Execution:** All evaluations run locally on your machine, giving you control over your data.
*   **Custom Metrics:** Build and integrate your own custom metrics seamlessly.
*   **Synthetic Dataset Generation:** Generate datasets for robust evaluation.
*   **CI/CD Integration:** Integrate DeepEval into any CI/CD environment.
*   **Red Teaming:**  Identify over 40 safety vulnerabilities (Toxicity, Bias, SQL Injection, etc.) with advanced attack strategies.
*   **LLM Benchmark Support:** Easily benchmark LLMs on popular benchmarks (MMLU, HellaSwag, DROP, etc.).
*   **Confident AI Integration:** Full lifecycle evaluation with [Confident AI](https://confident-ai.com?utm_source=GitHub) for dataset curation, benchmarking, metric tuning, debugging, monitoring, and more.

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

## 🔥 Metrics and Features

> 🥳 You can now share DeepEval's test results on the cloud directly on [Confident AI](https://confident-ai.com?utm_source=GitHub)'s infrastructure

- Supports both end-to-end and component-level LLM evaluation.
- Large variety of ready-to-use LLM evaluation metrics (all with explanations) powered by **ANY** LLM of your choice, statistical methods, or NLP models that runs **locally on your machine**:
  - G-Eval
  - DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
  - **RAG metrics:**
    - Answer Relevancy
    - Faithfulness
    - Contextual Recall
    - Contextual Precision
    - Contextual Relevancy
    - RAGAS
  - **Agentic metrics:**
    - Task Completion
    - Tool Correctness
  - **Others:**
    - Hallucination
    - Summarization
    - Bias
    - Toxicity
  - **Conversational metrics:**
    - Knowledge Retention
    - Conversation Completeness
    - Conversation Relevancy
    - Role Adherence
  - etc.
- Build your own custom metrics that are automatically integrated with DeepEval's ecosystem.
- Generate synthetic datasets for evaluation.
- Integrates seamlessly with **ANY** CI/CD environment.
- [Red team your LLM application](https://deepeval.com/docs/red-teaming-introduction) for 40+ safety vulnerabilities in a few lines of code, including:
  - Toxicity
  - Bias
  - SQL Injection
  - etc., using advanced 10+ attack enhancement strategies such as prompt injections.
- Easily benchmark **ANY** LLM on popular LLM benchmarks in [under 10 lines of code.](https://deepeval.com/docs/benchmarks-introduction?utm_source=GitHub), which includes:
  - MMLU
  - HellaSwag
  - DROP
  - BIG-Bench Hard
  - TruthfulQA
  - HumanEval
  - GSM8K
- [100% integrated with Confident AI](https://confident-ai.com?utm_source=GitHub) for the full evaluation lifecycle:
  - Curate/annotate evaluation datasets on the cloud
  - Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
  - Fine-tune metrics for custom results
  - Debug evaluation results via LLM traces
  - Monitor & evaluate LLM responses in product to improve datasets with real-world data
  - Repeat until perfection

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

## 🔌 Integrations

*   🦄 LlamaIndex:  [Unit test RAG applications in CI/CD](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 Hugging Face:  [Enable real-time evaluations during LLM fine-tuning](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br />

## 🚀 QuickStart

Get started with DeepEval by testing your RAG-based customer support chatbot.

### Installation

```bash
pip install -U deepeval
```

### Environment Variables

DeepEval auto-loads `.env.local` then `.env` from the current working directory at import time.
**Precedence:** process env -> `.env.local` -> `.env`.
Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

### Create an Account (Highly Recommended)

Create a free account on the DeepEval platform to store and share test results.

```bash
deepeval login
```

Follow the CLI instructions to create your account, copy your API key, and paste it into the CLI.

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Write a test case in `test_chatbot.py`:

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

Set your `OPENAI_API_KEY` environment variable:

```bash
export OPENAI_API_KEY="..."
```

Run the test:

```bash
deepeval test run test_chatbot.py
```

**Congratulations!**  Your test case should pass.  The `input` simulates a user query, `actual_output` is your app's response, and `expected_output` is the correct answer. `GEval` is a metric to evaluate your LLM output's correctness.

Read the [documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more details on metrics, custom metrics, and integrations.

<br />

## Evaluating Nested Components

Component-level evaluations help you test individual parts of your LLM application. Use the `@observe` decorator to apply metrics.

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

DeepEval metrics can be used independently.

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

## Bulk Evaluation of Datasets

Evaluate a dataset with multiple test cases:

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

Run tests:

```bash
# Run this in the CLI, you can also add an optional -n flag to run tests in parallel
deepeval test run test_<filename>.py -n 4
```

Alternative:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

## LLM Evaluation with Confident AI

The complete LLM evaluation lifecycle:

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

Learn more about Confident AI [here](https://www.confident-ai.com/docs?utm_source=GitHub).

Log in:

```bash
deepeval login
```

Run your tests again:

```bash
deepeval test run test_chatbot.py
```

View results in your browser.

![Demo GIF](assets/demo.gif)

<br />

## Configuration

### Environment variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

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

Licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).