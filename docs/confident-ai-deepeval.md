<p align="center">
    <img src="https://github.com/confident-ai/deepeval/blob/main/docs/static/img/deepeval.png" alt="DeepEval Logo" width="100%">
</p>

<h1 align="center">DeepEval: The Open-Source LLM Evaluation Framework</h1>

<p align="center">
  <a href="https://github.com/confident-ai/deepeval">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/confident-ai/deepeval?style=social">
  </a>
  <a href="https://discord.gg/3SEyvpgu2f">
      <img alt="Discord" src="https://img.shields.io/discord/1181738436242276423?label=Discord&logo=discord&style=social">
  </a>
  <a href="https://x.com/deepeval">
    <img alt="Twitter Follow" src="https://img.shields.io/twitter/follow/deepeval?style=social&logo=x">
  </a>
</p>

<p align="center">
    <a href="https://deepeval.com/docs/getting-started?utm_source=GitHub">Documentation</a> |
    <a href="#metrics-and-features">Metrics and Features</a> |
    <a href="#quickstart">Getting Started</a> |
    <a href="#integrations">Integrations</a> |
    <a href="https://confident-ai.com?utm_source=GitHub">DeepEval Platform</a>
</p>

DeepEval is your go-to open-source solution for evaluating and testing Large Language Model (LLM) systems, empowering you to build robust and reliable AI applications.

> [!IMPORTANT]
> [Sign up to the DeepEval platform](https://confident-ai.com?utm_source=GitHub) to compare iterations of your LLM app, generate & share testing reports, and more.

## Key Features:

*   **Comprehensive LLM Evaluation:** Supports both end-to-end and component-level evaluation.
*   **Extensive Metrics:** Includes a wide range of ready-to-use metrics, including G-Eval, RAG metrics, Agentic metrics, Conversational metrics and more, that run locally on your machine.
    *   **RAG Metrics:** Answer Relevancy, Faithfulness, Contextual Recall/Precision/Relevancy, RAGAS.
    *   **Agentic Metrics:** Task Completion, Tool Correctness
    *   **Additional Metrics:** Hallucination, Summarization, Bias, Toxicity, Knowledge Retention, Conversation Completeness/Relevancy/Role Adherence.
*   **Custom Metric Creation:** Build and integrate your own custom evaluation metrics seamlessly.
*   **Synthetic Dataset Generation:** Generate synthetic datasets for robust evaluation.
*   **CI/CD Integration:** Integrates effortlessly with any CI/CD environment.
*   **Red Teaming:** Easily red team your LLM applications with built in safety tests.
*   **Benchmarking:** Benchmark LLMs with ease on popular LLM benchmarks like MMLU, HellaSwag, and others.
*   **DeepEval Platform Integration:**  Seamlessly integrated with the Confident AI platform, providing a complete LLM evaluation lifecycle.

## Quickstart

Get started with DeepEval in minutes!

### Installation

```bash
pip install -U deepeval
```

### Creating an Account (Highly Recommended)

Using the `deepeval` platform allows you to generate shareable testing reports on the cloud. It is free, takes no additional code to setup, and we highly recommend giving it a try.

To login, run:

```bash
deepeval login
```

Follow the instructions in the CLI to create an account, copy your API key, and paste it into the CLI. All test cases will automatically be logged (find more information on data privacy [here](https://deepeval.com/docs/data-privacy?utm_source=GitHub)).

### Writing Your First Test Case

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

### Evaluating Nested Components

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

### Evaluating Without Pytest Integration

Alternatively, you can evaluate without Pytest, which is more suited for a notebook environment.

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

DeepEval is extremely modular, making it easy for anyone to use any of our metrics. Continuing from the previous example:

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

Note that some metrics are for RAG pipelines, while others are for fine-tuning. Make sure to use our docs to pick the right one for your use case.

### Evaluating a Dataset / Test Cases in Bulk

In DeepEval, a dataset is simply a collection of test cases. Here is how you can evaluate these in bulk:

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

## Integrations

*   🦄 LlamaIndex: [**Unit test RAG applications in CI/CD**](https://www.deepeval.com/integrations/frameworks/llamaindex?utm_source=GitHub)
*   🤗 Hugging Face: [**Enable real-time evaluations during LLM fine-tuning**](https://www.deepeval.com/integrations/frameworks/huggingface?utm_source=GitHub)

## LLM Evaluation With Confident AI

The correct LLM evaluation lifecycle is only achievable with [the DeepEval platform](https://confident-ai.com?utm_source=Github). It allows you to:

1.  Curate/annotate evaluation datasets on the cloud
2.  Benchmark LLM app using dataset, and compare with previous iterations to experiment which models/prompts works best
3.  Fine-tune metrics for custom results
4.  Debug evaluation results via LLM traces
5.  Monitor & evaluate LLM responses in product to improve datasets with real-world data
6.  Repeat until perfection

Everything on Confident AI, including how to use Confident is available [here](https://documentation.confident-ai.com/docs?utm_source=GitHub).

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

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](https://github.com/confident-ai/deepeval/blob/main/CONTRIBUTING.md) for details on our code of conduct and how to submit pull requests.

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

Built by the founders of Confident AI.  Contact jeffreyip@confident-ai.com for all enquiries.

## License

DeepEval is licensed under Apache 2.0 - see the [LICENSE.md](https://github.com/confident-ai/deepeval/blob/main/LICENSE.md) file for details.