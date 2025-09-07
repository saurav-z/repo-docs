<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<p align="center">
    <h1 align="center">DeepEval: Effortlessly Evaluate and Test Your LLM Applications</h1>
</p>

<p align="center">
<a href="https://trendshift.io/repositories/5917" target="_blank"><img src="https://trendshift.io/api/badge/repositories/5917" alt="confident-ai%2Fdeepeval | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

<p align="center">
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/3SEyvpgu2f?style=flat">
    </a>
</p>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval">
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

**DeepEval** is the open-source framework you need to rigorously evaluate and test your Large Language Model (LLM) applications. ([See the original repo](https://github.com/confident-ai/deepeval)).

<br/>

## Key Features:

*   **Comprehensive Metrics:**  Evaluate LLMs using a wide array of ready-to-use metrics, including:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   **RAG Metrics:** Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS.
    *   **Agentic Metrics:** Task Completion, Tool Correctness
    *   **Other Metrics:** Hallucination, Summarization, Bias, Toxicity
    *   **Conversational Metrics:** Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence
*   **Customization:** Easily build your own custom metrics and seamlessly integrate them into DeepEval's ecosystem.
*   **Synthetic Dataset Generation:**  Create synthetic datasets to enhance your evaluation process.
*   **CI/CD Integration:**  Seamlessly integrates with any CI/CD environment.
*   **Red Teaming:**  Red team your LLM applications for over 40 safety vulnerabilities with prompt injection techniques, including: Toxicity, Bias, SQL Injection.
*   **Benchmarking:** Quickly benchmark any LLM on popular benchmarks, including MMLU, HellaSwag, DROP, BIG-Bench Hard, TruthfulQA, HumanEval, GSM8K, and more.
*   **Confident AI Integration:** 100% integration with the [Confident AI](https://confident-ai.com?utm_source=GitHub) platform for the full evaluation lifecycle:
    *   Curate/annotate evaluation datasets on the cloud
    *   Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
    *   Fine-tune metrics for custom results
    *   Debug evaluation results via LLM traces
    *   Monitor & evaluate LLM responses in product to improve datasets with real-world data
    *   Repeat until perfection

>   **[Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub)** to compare iterations of your LLM app, generate & share testing reports, and more.

>   **[Join our Discord](https://discord.com/invite/3SEyvpgu2f)** to discuss LLM evaluation, get help with metrics, and connect with the community.

<br/>

## 🔌 Integrations

*   🦄 **LlamaIndex:**  [Unit test RAG applications in CI/CD](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub).
*   🤗 **Hugging Face:**  [Enable real-time evaluations during LLM fine-tuning](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub).

<br/>

## 🚀 Quickstart: Testing a RAG-based Chatbot

DeepEval makes it easy to evaluate your LLM applications.

### 1. Installation

```bash
pip install -U deepeval
```

### 2. Environment Variables

DeepEval loads environment variables from `.env.local` and then `.env` automatically.

**Precedence:** process env -> `.env.local` -> `.env`.

Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

### 3. Create an Account (Recommended)

Using the `deepeval` platform will allow you to generate sharable testing reports on the cloud.  It is free, takes no additional code to setup, and we highly recommend giving it a try.

To login, run:

```bash
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste it into the CLI.

### 4. Create a Test Case

Create a test file (e.g., `test_chatbot.py`):

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first test case:

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

### 5. Run Your Test

```bash
deepeval test run test_chatbot.py
```

**Congratulations! Your test case should have passed ✅**

### 6. Component-Level Evaluation

To evaluate individual components within your LLM app, use **component-level** evals. Trace components using the `@observe` decorator:

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

### 7. Evaluating Without Pytest

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

### 8. Using Standalone Metrics

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

### 9. Evaluating a Dataset

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

or

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

<br/>

## ☁️ LLM Evaluation with Confident AI

Achieve the complete LLM evaluation lifecycle with [the DeepEval platform](https://confident-ai.com?utm_source=Github):

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

To begin, login from the CLI:

```bash
deepeval login
```

<br/>

## Configuration

### Environment Variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br/>

## Contributing

Read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct and the contribution process.

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

Built by the founders of Confident AI. Contact jeffreyip@confident-ai.com for all enquiries.

<br/>

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.