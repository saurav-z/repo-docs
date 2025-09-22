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

## DeepEval: Supercharge Your LLM Testing with Open-Source Power

DeepEval is a cutting-edge, open-source framework designed to simplify the evaluation and testing of Large Language Model (LLM) applications, offering a comprehensive suite of tools to ensure accuracy and reliability; [learn more](https://github.com/confident-ai/deepeval).

**Key Features:**

*   **Comprehensive Metrics:** Evaluate LLMs with a diverse set of metrics, including G-Eval, RAG metrics (Answer Relevancy, Faithfulness, etc.), and agentic metrics (Task Completion, Tool Correctness), all running locally on your machine.
*   **Customizability:** Build your own custom metrics and seamlessly integrate them into the DeepEval ecosystem.
*   **CI/CD Integration:** Integrate DeepEval into any CI/CD environment for automated testing.
*   **Red Teaming:**  Test your LLM applications for over 40 safety vulnerabilities with features like toxicity, bias, and prompt injection detection.
*   **Benchmarking:** Easily benchmark LLMs on popular benchmarks like MMLU and HumanEval.
*   **Confident AI Integration:** Full integration with the [Confident AI platform](https://confident-ai.com?utm_source=GitHub) for dataset management, result comparison, debugging, and real-world performance monitoring.

>   Need a centralized hub for your DeepEval results and a collaborative platform for LLM testing? Sign up for the [DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.

![Demo GIF](assets/demo.gif)

> Want to discuss LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br />

## 🔑 Metrics and Features

*   **Supports End-to-End and Component-Level Evaluation:** Analyze your LLM applications at both the macro and micro levels.
*   **Ready-to-Use Metrics:** A wide range of pre-built LLM evaluation metrics powered by any LLM of your choice, statistical methods, or NLP models, running locally on your machine.  Includes:
    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG Metrics:** Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS
    *   **Agentic Metrics:** Task Completion, Tool Correctness
    *   **Other Metrics:** Hallucination, Summarization, Bias, Toxicity
    *   **Conversational Metrics:** Knowledge Retention, Conversation Completeness, Conversation Relevancy, Role Adherence
*   **Custom Metric Creation:** Easily create and integrate your own tailored evaluation metrics.
*   **Synthetic Dataset Generation:** Generate data for testing and analysis.
*   **CI/CD Compatibility:** Seamlessly integrate with existing CI/CD pipelines.
*   **Red Teaming Capabilities:** Test for vulnerabilities with advanced techniques (prompt injections).
*   **LLM Benchmark Support:** Evaluate LLMs on popular benchmarks within 10 lines of code.
*   **Full Confident AI Integration:**  Leverage Confident AI for a complete evaluation lifecycle.
    *   Curate/annotate evaluation datasets
    *   Benchmark LLM apps and compare iterations
    *   Fine-tune metrics
    *   Debug results with LLM traces
    *   Monitor and improve datasets with real-world data

>   **Note:** DeepEval's cloud platform is [Confident AI](https://app.confident-ai.com?utm_source=GitHub). Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

## 🔌 Integrations

*   🦄 **LlamaIndex:** Unit test RAG applications in CI/CD ([LlamaIndex Integration](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub))
*   🤗 **Hugging Face:** Enable real-time evaluations during LLM fine-tuning ([Hugging Face Integration](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub))

<br />

## 🚀 Quickstart

Get started with DeepEval quickly using a customer support chatbot example.

### Installation

```bash
pip install -U deepeval
```

### Account Creation (Highly Recommended)

Create a free account to store shareable test reports in the cloud:

```bash
deepeval login
```

Follow the CLI prompts to create an account and get your API key.

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Write your first test case:

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

Set your `OPENAI_API_KEY` as an environment variable.

```bash
export OPENAI_API_KEY="..."
```

Run your test:

```bash
deepeval test run test_chatbot.py
```

**Congratulations!** Your test case should pass.

*   `input`: User input.
*   `actual_output`: The LLM app's output.
*   `expected_output`: The ideal answer.
*   `GEval`: Research-backed metric to evaluate the output.
*   The metric scores range from 0 to 1, and the threshold determines if your test passed or not.

### Component-Level Evaluation

Evaluate specific components using the `@observe` decorator:

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

### Evaluating a Dataset

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

<br />

## ☁️ DeepEval With Confident AI

DeepEval's cloud platform, [Confident AI](https://confident-ai.com?utm_source=Github), allows you to:

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

Everything on Confident AI, including how to use Confident is available [here](https://www.confident-ai.com/docs?utm_source=GitHub).

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

## Configuration

### Environment Variables via .env files

Use `.env.local` or `.env` (optional) to store your environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br />

## Contributing

Read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for contribution guidelines.

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

Built by Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

<br />

## License

DeepEval is licensed under Apache 2.0; see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) for details.