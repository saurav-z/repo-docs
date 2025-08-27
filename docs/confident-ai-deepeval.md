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


## Evaluate, Test, and Improve Your LLMs with DeepEval

DeepEval is an open-source LLM evaluation framework, empowering you to rigorously test and enhance your Large Language Model applications.  [Explore the DeepEval Repo](https://github.com/confident-ai/deepeval) for comprehensive LLM testing and evaluation.

**Key Features:**

*   **Comprehensive Metrics:**  Evaluate LLM outputs with a diverse range of metrics, including G-Eval, RAG metrics (Answer Relevancy, Faithfulness, etc.), agentic metrics, and more. All metrics can be run locally!
*   **Component-Level Evaluation:** Easily test individual components within your LLM applications, such as LLM calls, retrievers, and agents, using the `@observe` decorator.
*   **Customization:** Build and integrate your own custom metrics seamlessly into the DeepEval ecosystem.
*   **Red Teaming:** Identify and mitigate safety vulnerabilities with built-in red-teaming capabilities, including toxicity, bias, and prompt injection detection.
*   **Benchmarking:**  Benchmark your LLMs against popular LLM benchmarks like MMLU, HellaSwag, and more, with minimal code.
*   **Integration with Confident AI:**  Leverage the full LLM evaluation lifecycle with our platform: curate datasets, compare iterations, debug results, and monitor performance in production.
*   **Flexible Integration:** Works with LangChain, LlamaIndex, and other frameworks.
*   **Easy to Use:** Test LLMs in just a few lines of code!

> [!IMPORTANT]
> Need a place for your DeepEval testing data to live 🏡❤️? [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.
>
> ![Demo GIF](assets/demo.gif)

> Want to talk LLM evaluation, need help picking metrics, or just to say hi? [Come join our discord.](https://discord.com/invite/3SEyvpgu2f)

<br />

## 🚀 Quickstart: Test your LLM app

Get started evaluating your LLM applications quickly with this simple example:

### 1. Installation
```bash
pip install -U deepeval
```

### 2. Create an Account (Highly Recommended)

Using the `deepeval` platform allows you to generate sharable testing reports.

To login, run:
```bash
deepeval login
```

Follow the instructions in the CLI.

### 3. Writing Your First Test Case
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
Set your `OPENAI_API_KEY` as an environment variable:

```
export OPENAI_API_KEY="..."
```

And finally, run `test_chatbot.py` in the CLI:

```
deepeval test run test_chatbot.py
```

### 4. Understanding the Results
*   `input`: mimics a user input.
*   `actual_output`: a placeholder for your application's output.
*   `expected_output`: the ideal answer for a given `input`.
*   `GEval`: a research-backed metric for evaluating your LLM's output.
*   Metric scores range from 0-1, determining if the test passed.

[Read our documentation](https://deepeval.com/docs/getting-started?utm_source=GitHub) for more details.

<br />

## 🔌 Integrations

*   🦄 [LlamaIndex](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 [Hugging Face](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

<br />

## 🔥 Metrics and Features

*   **Comprehensive LLM Evaluation:** Supports both end-to-end and component-level evaluation.
*   **Diverse Metric Selection:**
    *   **G-Eval**
    *   **DAG**
    *   **RAG Metrics**: Answer Relevancy, Faithfulness, Contextual Recall, Contextual Precision, Contextual Relevancy, RAGAS.
    *   **Agentic Metrics**: Task Completion, Tool Correctness
    *   **Other Metrics**: Hallucination, Summarization, Bias, Toxicity, Knowledge Retention, Conversation Completeness, etc.
*   **Custom Metric Creation:** Build and integrate custom metrics.
*   **Synthetic Dataset Generation:** Generate datasets for evaluation.
*   **CI/CD Integration:** Integrates seamlessly with CI/CD.
*   **Red Teaming Capabilities:**  Red team your LLM applications for safety vulnerabilities.
*   **LLM Benchmarking:** Easily benchmark LLMs on common benchmarks.
*   **[100% Integrated with Confident AI](https://confident-ai.com?utm_source=GitHub)**

> [!NOTE]
> Confident AI is the DeepEval platform. Create an account [here.](https://app.confident-ai.com?utm_source=GitHub)

<br />

## LLM Evaluation With Confident AI

The correct LLM evaluation lifecycle is only achievable with [the DeepEval platform](https://confident-ai.com?utm_source=Github). It allows you to:

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

## Contributing

See [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on how to contribute.

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

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file.