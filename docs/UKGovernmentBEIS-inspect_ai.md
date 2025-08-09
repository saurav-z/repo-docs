<!--
  Description: A framework for comprehensive large language model (LLM) evaluations, empowering researchers and developers to assess and improve AI performance.
  Keywords: LLM evaluation, AI safety, AI security, prompt engineering, model grading, tool usage, multi-turn dialogue, Python framework, UK AI Security Institute, Inspect, AI safety institute.
-->

# Inspect: Comprehensive LLM Evaluation Framework

**Safeguard AI with Inspect, a powerful framework from the UK AI Security Institute designed to meticulously evaluate and enhance the performance of large language models.**

[![Inspect Logo](https://inspect.aisi.org.uk/images/aisi-logo.svg)](https://aisi.gov.uk/)

Inspect, developed by the [UK AI Security Institute (AISI)](https://aisi.gov.uk/), offers a robust suite of tools to facilitate thorough evaluation of Large Language Models (LLMs). This framework is designed to empower researchers and developers in understanding and improving LLM behavior, promoting AI safety and security.

## Key Features

*   **Prompt Engineering:**  Design and refine prompts for optimal LLM performance.
*   **Tool Usage Analysis:**  Evaluate how effectively LLMs utilize tools and external resources.
*   **Multi-Turn Dialogue Support:**  Assess LLMs in complex, multi-turn conversational scenarios.
*   **Model Graded Evaluations:**  Implement sophisticated scoring and grading mechanisms for LLMs.
*   **Extensibility:**  Expand Inspect's capabilities with custom Python packages to support new elicitation and scoring techniques.

## Getting Started

For detailed information on using Inspect, please refer to the official documentation: [https://inspect.aisi.org.uk/](https://inspect.aisi.org.uk/)

## Development Setup

Contribute to Inspect by cloning the repository and installing the necessary dependencies:

```bash
git clone https://github.com/UKGovernmentBEIS/inspect_ai.git
cd inspect_ai
pip install -e ".[dev]"
```

### Optional Development Tools

*   **Pre-commit Hooks:** Install pre-commit hooks for code quality and consistency:

    ```bash
    make hooks
    ```

*   **Code Quality Checks:** Run linting, formatting, and tests to ensure code quality:

    ```bash
    make check
    make test
    ```

*   **VS Code Recommendations:**  If using VS Code, install the recommended extensions (Python, Ruff, and MyPy) when prompted.

**[View the original Inspect repository on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)**