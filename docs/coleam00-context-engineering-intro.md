# Supercharge Your AI Coding with Context Engineering

Context Engineering empowers your AI coding assistant to deliver complete, high-quality features, eliminating the limitations of prompt engineering.  Check out the [original repo here](https://github.com/coleam00/context-engineering-intro)!

**Key Features:**

*   **Comprehensive Context:** Provides AI with all necessary information for end-to-end feature implementation.
*   **Reduced Failures:** Minimizes AI errors by providing detailed guidelines, examples, and validation.
*   **Consistency & Standards:** Enforces project-specific patterns, conventions, and coding standards.
*   **Complex Feature Enablement:** Allows AI to manage multi-step processes and incorporate testing.
*   **Self-Correcting Capabilities:** Uses validation loops to identify and fix errors automatically.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Quick Start Guide](#quick-start)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering is a revolutionary approach to AI coding, offering a more robust and efficient alternative to prompt engineering.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:** Focuses on refining prompts to get the desired output.  Think of it like a sticky note: a quick reminder.

**Context Engineering:** Provides the AI with comprehensive, project-specific information, documentation, and validation. Think of it as a full, detailed screenplay.

### Why Context Engineering Matters

1.  **Minimize AI Failures:** Most agent failures stem from insufficient context, not model limitations.
2.  **Guarantee Consistency:** Ensures AI adheres to your project's code patterns and best practices.
3.  **Handle Complexity:** Empowers AI to manage multi-step implementations successfully.
4.  **Enable Self-Correction:** Integrated validation loops allow the AI to fix its own issues.

## Quick Start

Get started with Context Engineering using these quick steps:

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set up project rules (optional)
# Customize the CLAUDE.md file for your project guidelines

# 3. Add code examples (recommended)
# Place relevant code examples in the examples/ directory

# 4. Create a feature request
# Edit the INITIAL.md file with your specific requirements

# 5. Generate a comprehensive PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code, run:
/execute-prp PRPs/your-feature-name.md
```

## Template Structure

The Context Engineering template has the following structure:

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

The template provides the necessary structure for implementing Context Engineering and setting up your coding environment.

## Step-by-Step Guide

Follow these steps to begin using the Context Engineering template:

### 1. Set Up Global Rules (CLAUDE.md)

Customize the `CLAUDE.md` file to define project-wide rules that your AI assistant will follow:

*   **Project Understanding:** Project awareness, task verification.
*   **Code Formatting:** File size, module organization guidelines.
*   **Testing Methods:** Unit test patterns, test coverage details.
*   **Style Standards:** Preferred language and formatting rules.
*   **Documentation:** Docstring and commenting standards.

You can adapt the provided template to match your project's unique needs.

### 2. Create Your Initial Feature Request

Edit the `INITIAL.md` file to detail the feature you want to implement:

```markdown
## FEATURE:
[Describe what you want to build, specify functionality and requirements]

## EXAMPLES:
[List example files in the examples/ folder and their purpose]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or resources]

## OTHER CONSIDERATIONS:
[Mention any requirements or things the AI may miss]
```

Refer to `INITIAL_EXAMPLE.md` for a complete example of a feature request.

### 3. Generate the PRP

PRPs (Product Requirements Prompts) are comprehensive documents that guide the AI through the implementation process:

*   Comprehensive context and documentation.
*   Step-by-step implementation guidelines with validation.
*   Error handling protocols.
*   Testing requirements.

Run the following command within Claude Code:

```bash
/generate-prp INITIAL.md
```

This command does the following:

1.  Reads your feature request.
2.  Analyzes your codebase.
3.  Searches for supporting documentation.
4.  Generates a PRP in the `PRPs/` directory.

### 4. Execute the PRP

Once the PRP is generated, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Review the PRP context.
2.  Create a detailed implementation plan.
3.  Execute each step while validating at each point.
4.  Run tests and make fixes.
5.  Confirm that all success criteria are met.

## Writing Effective INITIAL.md Files

Optimize your `INITIAL.md` files for the best results:

### Key Sections Explained

**FEATURE**: Be precise about your requirements.
    *   ❌ "Create a web scraper"
    *   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Use examples/ folder.
    *   Place relevant code patterns in `examples/`.
    *   Reference specific files and patterns.
    *   Explain how to mimic the examples.

**DOCUMENTATION**: Integrate your resources.
    *   Include API documentation URLs.
    *   Include library guides.
    *   Include database schemas.

**OTHER CONSIDERATIONS**: Include critical details.
    *   Consider authentication needs.
    *   Specify rate limits and quotas.
    *   Address common problems.
    *   Describe performance requirements.

## The PRP Workflow

Understand how PRPs work behind the scenes:

### How /generate-prp Works

The `/generate-prp` command goes through a process to create effective Product Requirements Prompts:

1.  **Research Phase**: Analyzes your codebase, looks for patterns, and identifies conventions.
2.  **Documentation Gathering**: Gathers documentation from APIs, libraries, and addresses any potential quirks.
3.  **Blueprint Creation**: Creates a step-by-step implementation plan, includes validation checks, and defines the testing criteria.
4.  **Quality Assurance**: Confirms all required context is incorporated and provides a confidence rating.

### How /execute-prp Works

1.  **Load Context**: Reads all information from the PRP.
2.  **Plan**: Develops a detailed task list using the TodoWrite method.
3.  **Execute**: Completes each task by implementing all necessary parts.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Makes fixes as needed.
6.  **Complete**: Verifies that all criteria are met.

## Using Examples Effectively

The `examples/` folder is essential for successful implementations. The AI will understand your patterns better when they can reference code.

### What to Include in Examples

1.  **Code Structure Patterns**: Show how you organize modules, import conventions, and patterns for classes and functions.

2.  **Testing Patterns**: Show your test file structure, approaches for mocking, and how you want your assertions structured.

3.  **Integration Patterns**: Show examples of API client implementations, database connections, and authentication flows.

4.  **CLI Patterns**: Examples of how to implement argument parsing, output formatting, and error handling.

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

Follow these best practices for optimal results:

### 1. Be Explicit in INITIAL.md

*   Provide clear requirements and constraints.
*   Reference relevant examples.

### 2. Provide Comprehensive Examples

*   The more examples, the better the implementation.
*   Show what should and shouldn't be done, including error-handling patterns.

### 3. Use Validation Gates

*   PRPs include test commands that must pass.
*   Ensure working code the first time through.

### 4. Leverage Documentation

*   Include official API documents.
*   Reference specific sections of relevant documentation.

### 5. Customize CLAUDE.md

*   Integrate your conventions.
*   Define project-specific rules and coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)