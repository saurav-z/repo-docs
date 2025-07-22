# Context Engineering Template: Supercharge Your AI Coding with Comprehensive Context

**Unlock the power of AI coding assistants with Context Engineering – providing the context your AI needs to deliver superior, consistent, and complex solutions.**

[View the Original Repo on GitHub](https://github.com/coleam00/context-engineering-intro)

## Key Features

*   **Comprehensive Context:** Provides AI with all necessary information for end-to-end solutions.
*   **Reduced AI Failures:** Minimizes errors by addressing context failures.
*   **Consistent Output:** Ensures AI adheres to project patterns, conventions, and standards.
*   **Simplified Complex Tasks:** Enables AI to handle multi-step implementations effectively.
*   **Self-Correcting Mechanism:** Utilizes validation loops for automated error correction.
*   **PRP Workflow:** Streamlined process for generating and executing Product Requirement Prompts.
*   **Example-Driven Development:** Leverages examples to guide the AI's coding behavior.
*   **Customizable:** Easily adaptable to project-specific rules and conventions.

## Getting Started

```bash
# 1. Clone the template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Configure project rules (optional)
# Edit CLAUDE.md to define your project's coding guidelines

# 3. Add code examples (essential)
# Place code examples in the examples/ directory to provide reference for AI

# 4. Create feature requests
# Describe your desired features in INITIAL.md

# 5. Generate a comprehensive Product Requirements Prompt (PRP)
# In Claude Code:
/generate-prp INITIAL.md

# 6. Execute the PRP to implement your feature
# In Claude Code:
/execute-prp PRPs/your-feature-name.md
```

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering elevates AI coding beyond prompt engineering, providing a comprehensive framework to guide AI assistants.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting specific prompts.
*   Limited in its ability to provide context.
*   Analogous to providing a sticky note with instructions.

**Context Engineering:**

*   Provides a complete system with context and documentation.
*   Encompasses documentation, examples, rules, and validation.
*   Comparable to writing a complete screenplay with all details.

### Why Context Engineering Matters

1.  **Reduced AI Failures:** Addresses context-related failures for enhanced reliability.
2.  **Consistent Output:** Ensures the AI follows project standards and conventions.
3.  **Enables Complex Features:** Allows AI to handle intricate, multi-step tasks.
4.  **Self-Correcting:** Utilizes validation loops for automated issue resolution.

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

*Note: Focus is on using context engineering to enhance code generation and not RAG or tools.

## Step-by-Step Guide

### 1. Set Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules for the AI assistant, including:

*   Project awareness
*   Code structure guidelines
*   Testing requirements
*   Style conventions
*   Documentation standards

**Customize the provided template to align with your project's needs.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to specify the feature you want to build. Use the following format:

```markdown
## FEATURE:
[Describe your desired functionality and any requirements.]

## EXAMPLES:
[List code examples in the examples/ directory and explain their use.]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, and other resources.]

## OTHER CONSIDERATIONS:
[Mention any special requirements, known issues, or common AI mistakes.]
```

**Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.**

### 3. Generate the PRP

Product Requirements Prompts (PRPs) are comprehensive implementation plans.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

This command will:

1.  Analyze your codebase and search for patterns.
2.  Gather relevant documentation.
3.  Create a detailed PRP in the `PRPs/` directory.

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

### 4. Execute the PRP

Execute the generated PRP to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Create an implementation plan.
3.  Execute each step with validation.
4.  Run tests and fix any issues.
5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Provide a detailed, specific description of the desired functionality.

*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Leverage code examples in the `examples/` directory.

*   Reference specific files and patterns.
*   Explain how they should be used.

**DOCUMENTATION**: Include all relevant resources.

*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS**: Capture any crucial details.

*   Authentication
*   Rate limits
*   Known issues
*   Performance requirements

## The PRP Workflow

### How `/generate-prp` Works

The command performs the following steps:

1.  **Research Phase:** Analyzes your codebase and identifies patterns.
2.  **Documentation Gathering:** Fetches and incorporates relevant documentation.
3.  **Blueprint Creation:** Generates a step-by-step implementation plan with validation steps and test requirements.
4.  **Quality Check:** Scores the confidence level and validates context inclusion.

### How `/execute-prp` Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates a detailed task list
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any identified issues
6.  **Complete**: Ensures all requirements are fulfilled

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example.

## Using Examples Effectively

The `examples/` folder is **crucial** for guiding the AI's behavior.

### What to Include in Examples

1.  **Code Structure Patterns:** Module organization, import conventions, class and function structures.

2.  **Testing Patterns:** Test file structure, mocking techniques, and assertion styles.

3.  **Integration Patterns:** API client implementations, database connections, and authentication flows.

4.  **CLI Patterns:** Argument parsing, output formatting, and error handling.

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

*   Be specific about requirements.
*   Reference examples as guidance.

### 2. Provide Comprehensive Examples

*   More examples lead to better implementations.
*   Show both what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs use test commands to validate the code.
*   The AI iterates until all validations are successful.

### 4. Leverage Documentation

*   Include official API docs.
*   Include other relevant resources.

### 5. Customize CLAUDE.md

*   Add project-specific rules.
*   Define your coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)