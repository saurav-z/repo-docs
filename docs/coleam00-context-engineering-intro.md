# Context Engineering: Supercharge Your AI Coding Assistant

**Unlock unparalleled efficiency and precision in your coding projects with Context Engineering, a revolutionary approach that goes beyond prompt engineering.** (Link to Original Repo: [https://github.com/coleam00/context-engineering-intro](https://github.com/coleam00/context-engineering-intro))

**Key Features:**

*   **Comprehensive Context:** Provide your AI with everything it needs to succeed, from documentation and examples to project rules and validation.
*   **Reduced AI Failures:** Minimize errors by giving your AI the complete picture, resulting in more reliable and consistent code generation.
*   **Project-Specific Consistency:** Ensure your AI adheres to your coding standards and project conventions.
*   **Enable Complex Features:** Empower your AI to handle multi-step implementations with confidence.
*   **Self-Correcting Capabilities:** Leverage validation loops to automatically fix errors and ensure high-quality output.

## Getting Started

This template provides a robust framework for implementing Context Engineering.

**1. Clone the Repository**

```bash
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro
```

**2. Set Up Project Rules (Optional)**

*   Customize the `CLAUDE.md` file to establish project-specific guidelines, including code structure, testing, and style conventions.

**3. Include Code Examples (Highly Recommended)**

*   Place relevant code examples in the `examples/` folder to guide your AI's implementation.

**4. Create Feature Requests**

*   Edit `INITIAL.md` files to clearly define your feature requirements.

**5. Generate a Product Requirements Prompt (PRP)**

*   Use the command: `/generate-prp INITIAL.md` within Claude Code to generate a detailed implementation blueprint.

**6. Execute the PRP**

*   Use the command: `/execute-prp PRPs/your-feature-name.md` within Claude Code to execute the PRP and implement your feature.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
    *   [Prompt Engineering vs. Context Engineering](#prompt-engineering-vs-context-engineering)
    *   [Why Context Engineering Matters](#why-context-engineering-matters)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering surpasses traditional prompt engineering by providing AI coding assistants with complete information.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Relies on clever wording and phrasing.
*   Limited by the task's phrasing.
*   Like a simple note.

**Context Engineering:**
*   A complete system for providing comprehensive context.
*   Includes all necessary information, from documentation and examples to rules and validation.
*   Like writing a comprehensive screenplay.

### Why Context Engineering Matters

1.  **Reduce AI Failures:** Address context-related failures instead of model limitations.
2.  **Ensure Consistency:** Guarantee adherence to project patterns and conventions.
3.  **Enable Complex Features:** Facilitate multi-step implementations with proper context.
4.  **Self-Correcting:** Utilize validation loops for automated error correction.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates comprehensive PRPs
│   │   └── execute-prp.md     # Executes PRPs to implement features
│   └── settings.local.json    # Claude Code permissions
├── PRPs/
│   ├── templates/
│   │   └── prp_base.md       # Base template for PRPs
│   └── EXAMPLE_multi_agent_prp.md  # Example of a complete PRP
├── examples/                  # Your code examples (critical!)
├── CLAUDE.md                 # Global rules for AI assistant
├── INITIAL.md               # Template for feature requests
├── INITIAL_EXAMPLE.md       # Example feature request
└── README.md                # This file
```

## Step-by-Step Guide

Follow these steps to implement Context Engineering in your projects:

### 1. Configure Global Rules (CLAUDE.md)

*   The `CLAUDE.md` file establishes project-wide rules.
*   Customize the template to include code structure, testing, style, and documentation standards.

### 2. Create Your Initial Feature Request

*   Edit `INITIAL.md` to specify feature requirements.
*   Include:
    *   Feature description
    *   Relevant examples
    *   Links to documentation
    *   Additional considerations

### 3. Generate the PRP

*   Run `/generate-prp INITIAL.md` in Claude Code.
*   The command analyzes your codebase, gathers documentation, and creates a PRP.

### 4. Execute the PRP

*   Run `/execute-prp PRPs/your-feature-name.md` in Claude Code to execute the PRP.
*   The AI assistant will implement the feature, validate it, and iterate until all requirements are met.

## Writing Effective INITIAL.md Files

Follow these guidelines for creating effective `INITIAL.md` files:

### Key Sections Explained

**FEATURE:**
*   Describe the desired functionality and all the requirements in detail.
*   **Example:** *"Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."*

**EXAMPLES:**
*   Reference relevant code examples from the `examples/` folder.
*   Explain how the examples should be used.

**DOCUMENTATION:**
*   Include all relevant resources, such as API documentation, library guides, and database schemas.

**OTHER CONSIDERATIONS:**
*   Specify authentication requirements, rate limits, or common pitfalls.

## The PRP Workflow

Understand how `/generate-prp` and `/execute-prp` work:

### How /generate-prp Works

1.  **Research Phase:** Analyze the codebase and identify patterns.
2.  **Documentation Gathering:** Fetch relevant documentation.
3.  **Blueprint Creation:** Create a step-by-step implementation plan.
4.  **Quality Check:** Score confidence level and include all the necessary context.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests.
5.  **Iterate:** Fixes any issues.
6.  **Complete:** Ensures all requirements are met.

## Using Examples Effectively

The `examples/` folder is crucial for success. Well-crafted examples help AI coding assistants learn and apply your coding patterns effectively.

### What to Include in Examples

1.  Code Structure Patterns
2.  Testing Patterns
3.  Integration Patterns
4.  CLI Patterns

### Example Structure

```
examples/
├── README.md           # Explains what each example demonstrates
├── cli.py             # CLI implementation pattern
├── agent/             # Agent architecture patterns
│   ├── agent.py      # Agent creation pattern
│   ├── tools.py      # Tool implementation pattern
│   └── providers.py  # Multi-provider pattern
└── tests/            # Testing patterns
    ├── test_agent.py # Unit test patterns
    └── conftest.py   # Pytest configuration
```

## Best Practices

### 1. Be Explicit in INITIAL.md
*   Provide specific requirements and constraints.
*   Reference examples.

### 2. Provide Comprehensive Examples
*   Use numerous examples to show your preferred coding patterns.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands that must pass.
*   This ensures working code on the first try.

### 4. Leverage Documentation
*   Include API docs.
*   Reference documentation sections.

### 5. Customize CLAUDE.md
*   Add your conventions.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)