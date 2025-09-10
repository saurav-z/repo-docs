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

## 🤖 DeepEval: Effortlessly Evaluate and Test Your LLMs

**DeepEval is an open-source LLM evaluation framework that simplifies the process of testing and optimizing your large language model applications.** This framework offers a comprehensive suite of tools for assessing LLM performance, ensuring accuracy, and preventing issues such as hallucinations and bias. Visit the [DeepEval GitHub repository](https://github.com/confident-ai/deepeval) to learn more.

<br/>

### Key Features:

*   ✅ **Comprehensive Metrics:** Evaluate LLMs with various ready-to-use metrics:
    *   G-Eval
    *   DAG (Deep Acyclic Graph)
    *   RAG metrics: Answer Relevancy, Faithfulness, Contextual Recall, Precision, Relevancy, RAGAS
    *   Agentic metrics: Task Completion, Tool Correctness
    *   Others: Hallucination, Summarization, Bias, Toxicity
    *   Conversational metrics: Knowledge Retention, Conversation Completeness, Relevancy, Role Adherence
*   ⚙️ **Custom Metric Creation:** Build and seamlessly integrate your custom evaluation metrics.
*   🧪 **Synthetic Dataset Generation:** Create synthetic datasets for robust LLM testing.
*   🔄 **CI/CD Integration:** Integrates with any CI/CD environment.
*   🛡️ **Red Teaming:** Red team your LLM application with over 40 safety vulnerabilities in a few lines of code.
*   📊 **Benchmarking:** Easily benchmark LLMs on popular benchmarks like MMLU, HellaSwag, and more.
*   ☁️ **Confident AI Integration:** Full lifecycle support with the [Confident AI](https://confident-ai.com?utm_source=GitHub) platform.

<br/>

### 🔌 Integrations

*   🦄 **LlamaIndex**: Unit test RAG applications in CI/CD.
*   🤗 **Hugging Face**: Enable real-time evaluations during LLM fine-tuning.

<br/>

### 🚀 Quickstart Guide

DeepEval offers both end-to-end and component-level evaluation. Below is a quickstart example, more detailed guides can be found in the documentation.

1.  **Installation:**

    ```bash
    pip install -U deepeval
    ```

2.  **Environment Variables:**
    ```bash
    cp .env.example .env.local
    # then edit .env.local (ignored by git)
    ```

3.  **Login and Account Creation (Highly Recommended):**
    ```bash
    deepeval login
    ```

    Follow the CLI prompts to create an account and retrieve your API key.

4.  **Writing your first test case**

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

### Component-Level Evaluations

If you wish to evaluate individual components within your LLM app, you need to run **component-level** evals - a powerful way to evaluate any component within an LLM system.

Simply trace "components" such as LLM calls, retrievers, tool calls, and agents within your LLM application using the `@observe` decorator to apply metrics on a component-level. Tracing with `deepeval` is non-instrusive (learn more [here](https://deepeval.com/docs/evaluation-llm-tracing#dont-be-worried-about-tracing)) and helps you avoid rewriting your codebase just for evals:

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

You can learn everything about component-level evaluations [here.](https://www.deepeval.com/docs/evaluation-component-level-llm-evals)

<br/>

### Other ways to evaluate

There are other ways to evaluate depending on your needs:
*   Evaluating Without Pytest Integration.
*   Using Standalone Metrics
*   Evaluating a Dataset / Test Cases in Bulk

Refer to the original README for example code for each of the above.

<br/>

## LLM Evaluation With Confident AI

**Unlock the full potential of LLM evaluation with [the DeepEval platform](https://confident-ai.com?utm_source=Github).**

*   Curate/annotate evaluation datasets
*   Benchmark and compare LLM apps
*   Fine-tune metrics
*   Debug evaluation results
*   Monitor and evaluate LLM responses in production
*   Iterate and improve

<br/>

### Configuration

*   **Environment variables via .env files:**

    *   `DEEPEVAL_DISABLE_DOTENV=1` to disable

    **Precedence:** process env -> `.env.local` -> `.env`

<br/>

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details.

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

Developed by the founders of Confident AI. Contact jeffreyip@confident-ai.com for inquiries.

<br/>

## License

DeepEval is licensed under Apache 2.0 - see [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md).