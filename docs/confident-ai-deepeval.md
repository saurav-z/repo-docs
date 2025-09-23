<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>

<p align="center">
    <a href="https://github.com/confident-ai/deepeval">
        <img src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social" alt="GitHub Stars">
    </a>
    <a href="https://discord.gg/3SEyvpgu2f">
        <img alt="discord-invite" src="https://dcbadge.vercel.app/api/server/3SEyvpgu2f?style=flat">
    </a>
</p>

<p align="center">
    <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
    <a href="#metrics-and-features">Metrics & Features</a> |
    <a href="#quickstart">Quickstart</a> |
    <a href="#integrations">Integrations</a> |
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

**Tired of LLMs that don't perform?**  DeepEval is a powerful, open-source framework for evaluating and testing Large Language Models (LLMs), enabling you to build robust and reliable AI applications.  Built to be like Pytest, DeepEval specializes in unit testing LLM outputs, incorporating the latest research to provide comprehensive evaluation metrics.

*   **Key Features:**

    *   **Comprehensive Metrics:** Evaluate LLMs with a variety of metrics, including G-Eval, answer relevancy, hallucination detection, RAGAS, and more. Metrics run **locally on your machine** using LLMs and NLP models.
    *   **Flexible Evaluation:**  Supports end-to-end and component-level evaluation, allowing you to test entire LLM applications or specific components.
    *   **Custom Metrics:** Easily create and integrate your own custom metrics into the DeepEval ecosystem.
    *   **Synthetic Dataset Generation:**  Generate synthetic datasets for robust evaluation and testing.
    *   **CI/CD Integration:** Seamlessly integrates with any CI/CD environment for automated testing.
    *   **Red Teaming Capabilities:**  Red team your LLM applications with pre-built tests for over 40 safety vulnerabilities.
    *   **Benchmarking:** Benchmark LLMs on popular benchmarks with minimal code.
    *   **Confident AI Integration:** 100% Integrated with Confident AI for the full evaluation lifecycle: from dataset curation to monitoring in production
*   **Use Cases:**
    *   Improve RAG pipelines
    *   Optimize agentic workflows
    *   Prevent prompt drift
    *   Confidently migrate from OpenAI to self-hosting LLMs
*   **Integrations:**
    *   [LlamaIndex](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub) - Unit test RAG applications.
    *   [Hugging Face](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub) - Real-time evaluations during fine-tuning.

> [!IMPORTANT]
> **Elevate your LLM evaluation with the DeepEval platform!** [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations, generate and share reports, and more.

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br/>

## <a id="metrics-and-features">🔥 Metrics and Features</a>

*   **End-to-End and Component-Level Evaluation:** Test your LLM applications at various levels of granularity.
*   **Ready-to-Use Metrics:** Utilize a wide array of pre-built metrics, powered by your choice of LLM or NLP models running locally:
    *   G-Eval
    *   DAG ([deep acyclic graph](https://deepeval.com/docs/metrics-dag))
    *   **RAG Metrics:** Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS
    *   **Agentic Metrics:** Task Completion, Tool Correctness
    *   **Other Metrics:** Hallucination, Summarization, Bias, Toxicity, Conversational metrics, etc.
*   **Custom Metric Creation:** Build your own custom metrics that integrate seamlessly with DeepEval.
*   **Dataset Generation:** Generate synthetic datasets for comprehensive evaluation.
*   **CI/CD Integration:** Integrates seamlessly with ANY CI/CD environment.
*   **Red Teaming:** Red team your LLM applications with a multitude of safety tests, including:
    *   Toxicity
    *   Bias
    *   SQL Injection
    *   And much more.
*   **LLM Benchmarking:** Easily benchmark LLMs on popular LLM benchmarks, including:
    *   MMLU
    *   HellaSwag
    *   DROP
    *   BIG-Bench Hard
    *   TruthfulQA
    *   HumanEval
    *   GSM8K
*   **Confident AI Integration:** Seamlessly integrated with Confident AI for the full LLM lifecycle:
    *   Dataset curation
    *   Model comparison
    *   Metric fine-tuning
    *   Trace debugging
    *   Response monitoring
    *   Iteration and improvement.

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br/>

## <a id="integrations">🔌 Integrations</a>

*   **LlamaIndex:** Unit test RAG applications within your CI/CD pipeline.
*   **Hugging Face:** Enable real-time evaluations during LLM fine-tuning.

<br/>

## <a id="quickstart">🚀 Quickstart</a>

Get started evaluating your LLM application quickly.

### Installation

DeepEval requires **Python>=3.9+**.

```bash
pip install -U deepeval
```

### Create an Account (Highly Recommended)

Using the DeepEval platform ([Confident AI](https://confident-ai.com?utm_source=GitHub)) allows you to generate and share your test results in the cloud. It's free, requires minimal setup, and offers valuable features.

To login, run:

```bash
deepeval login
```

Follow the CLI instructions to create or access your account.  Enter your API key when prompted.

### Writing Your First Test Case

Create a test file:

```bash
touch test_chatbot.py
```

Open `test_chatbot.py` and write your first end-to-end test case:

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

Set your `OPENAI_API_KEY` as an environment variable (or use your custom model; more details in the [documentation](https://deepeval.com/docs/metrics-introduction#using-a-custom-llm?utm_source=GitHub)):

```bash
export OPENAI_API_KEY="..."
```

Run the test from the command line:

```bash
deepeval test run test_chatbot.py
```

**Congratulations! Your test should have passed ✅**

*   `input`:  User input.
*   `actual_output`:  Your LLM application's output.
*   `expected_output`: The ideal answer.
*   `GEval`: A research-backed metric for evaluating your LLM's output.
*   `criteria`: Correctness of the `actual_output` relative to the `expected_output`.
*   Metric scores range from 0-1, with the `threshold` determining pass/fail.

[Read the documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more information.

### Evaluating Nested Components

Use component-level evaluations to test individual parts of your LLM application, such as LLM calls or retrievers.  Use the `@observe` decorator:

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

### Evaluating Without Pytest Integration

Evaluate in a notebook environment:

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

DeepEval's modular design allows you to use individual metrics.

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

Evaluate a dataset without pytest integration:

```python
from deepeval import evaluate
...

evaluate(dataset, [answer_relevancy_metric])
# or
dataset.evaluate([answer_relevancy_metric])
```

### Environment Variables (.env / .env.local)

DeepEval loads `.env.local` then `.env` at import time.

**Precedence:** process env -> `.env.local` -> `.env`.

Opt out with `DEEPEVAL_DISABLE_DOTENV=1`.

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)
```

<br/>

## DeepEval with Confident AI

With [Confident AI](https://confident-ai.com?utm_source=Github), you can:

1.  Curate/annotate evaluation datasets.
2.  Benchmark and compare models.
3.  Fine-tune metrics.
4.  Debug evaluation results.
5.  Monitor LLM responses in production.
6.  Iterate and improve.

Log in from the CLI:

```bash
deepeval login
```

After testing is finished, a link is displayed in the CLI. Paste it into your browser to view the results!

<br/>

## Configuration

### Environment variables via .env files

Using `.env.local` or `.env` is optional. If they are missing, DeepEval uses your existing environment variables. When present, dotenv environment variables are auto-loaded at import time (unless you set `DEEPEVAL_DISABLE_DOTENV=1`).

**Precedence:** process env -> `.env.local` -> `.env`

```bash
cp .env.example .env.local
# then edit .env.local (ignored by git)

<br />

## Contributing

Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details.

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

Built by the founders of Confident AI.  Contact jeffreyip@confident-ai.com for inquiries.

<br/>

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file.
```
Key improvements and SEO optimization in this version:

*   **Concise Hook:** A strong opening sentence grabs the reader's attention immediately.
*   **Clear Headings:**  Uses `h1`, `h2`, and more for better structure and readability, crucial for SEO.
*   **Keyword Optimization:** The use of relevant keywords like "LLM evaluation," "Large Language Models," "open-source," and metric names.
*   **Bulleted Lists:**  Easy-to-scan bullet points for features, use cases, and benefits, which helps with readability and SEO.
*   **Internal Linking:** Includes links to sections within the README and external links to other relevant pages, increasing time on page and SEO.
*   **Emphasis on Benefits:**  Highlights the advantages of using DeepEval (robustness, reliability, etc.).
*   **Call to Action:** Includes a clear call to action, encouraging users to sign up for the DeepEval platform.
*   **Concise Summaries:**  Replaces long paragraphs with shorter, more digestible descriptions.
*   **Clean formatting**: Used markdown and HTML appropriately to format the documentation.
*   **Focus on Value:** The README directly communicates the *value* DeepEval provides.