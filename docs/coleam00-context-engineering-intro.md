# Supercharge Your AI Coding with Context Engineering

**Context Engineering elevates AI coding assistants from basic prompt responders to powerful, context-aware collaborators.** This template provides a streamlined approach to building sophisticated AI-powered coding workflows.  Learn more and get started at the [original repo](https://github.com/coleam00/context-engineering-intro).

**Key Features:**

*   **Context-Driven AI:**  Move beyond basic prompts by providing comprehensive context including documentation, examples, and project-specific rules.
*   **Streamlined Workflow:** Utilize custom slash commands to generate and execute Product Requirements Prompts (PRPs) for automated feature implementation.
*   **Enhanced Accuracy & Consistency:**  Reduce AI failures and ensure the AI assistant adheres to your project's patterns and conventions.
*   **Comprehensive Template:**  Includes a clear structure and examples for setting up your project and guiding the AI assistant.
*   **Self-Correcting Implementations:** Leverage validation loops that allows AI assistants to fix their own mistakes.

## Getting Started

```bash
# 1. Clone this template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (Optional - Start with the Template)
# Edit CLAUDE.md to define your project's guidelines

# 3. Incorporate Code Examples (Highly Recommended)
# Place relevant code examples in the examples/ folder

# 4. Define a Feature Request
# Edit INITIAL.md with your feature requirements

# 5. Generate a PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement Your Feature
# In Claude Code, run:
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

Context Engineering is a comprehensive methodology that moves past the limitations of prompt engineering to provide AI assistants with all the information needed to complete coding tasks accurately and efficiently.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting the perfect prompt.
*   Limited by the phrasing of the prompt.
*   Equivalent to providing a sticky note with instructions.

**Context Engineering:**

*   Provides a full system for supplying comprehensive context.
*   Includes documentation, examples, rules, and validation steps.
*   Similar to providing a full screenplay with all the details.

### Why Context Engineering is Essential

1.  **Reduces AI Failures**: Most failures result from a lack of context.
2.  **Ensures Consistency**: Your project's patterns and conventions will be followed.
3.  **Enables Complex Features**: AI can create multi-step features.
4.  **Self-Correcting**: Includes validation loops that allow AI to fix errors.

## Template Structure

```
context-engineering-intro/
├── .claude/
│   ├── commands/
│   │   ├── generate-prp.md    # Generates PRPs
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

### 1.  Set Up Project Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules for the AI assistant. Key aspects include:

*   **Project Awareness**: Understanding project goals and tasks.
*   **Code Structure**:  Adhering to code structure guidelines.
*   **Testing Requirements**: Following unit testing patterns and coverage goals.
*   **Style Conventions**: Using desired language, format rules and coding standards.
*   **Documentation Standards**:  Using documentation formats and commenting practices.

**Customize the provided template to match your project's requirements.**

### 2. Create a Feature Request (INITIAL.md)

Edit `INITIAL.md` to detail what you want the AI to build:

```markdown
## FEATURE:
[Be specific - describe functionality and all requirements]

## EXAMPLES:
[List any example files and explain how they should be used]

## DOCUMENTATION:
[Include links to all related documentation and resources]

## OTHER CONSIDERATIONS:
[Mention potential issues, requirements, or things the AI might overlook]
```

**See `INITIAL_EXAMPLE.md` for a complete example.**

### 3. Generate the PRP (Product Requirements Prompt)

PRPs (Product Requirements Prompts) are implementation blueprints:

*   Provide complete context and documentation.
*   Include implementation steps and validations.
*   Include error-handling patterns.
*   Detail the test requirements.

Run this command in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are defined in `.claude/commands/`.

This command will:

1.  Read your feature request.
2.  Analyze the codebase.
3.  Find relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once generated, run this command in Claude Code:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read all context from the PRP.
2.  Develop an implementation plan.
3.  Execute each step and validate.
4.  Run tests and fix errors.
5.  Ensure all requirements are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Be specific and comprehensive.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**:  Use the `/examples/` folder.
*   Add related code patterns in `/examples/`.
*   Reference specific files and patterns.
*   Explain what to replicate.

**DOCUMENTATION**:  Include all relevant resources.
*   Include API documentation URLs.
*   Include library guides.
*   Include resources for server documentation.
*   Include database schemas.

**OTHER CONSIDERATIONS**:  Include important details.
*   Detail authentication requirements.
*   Note rate limits or quotas.
*   Detail common pitfalls.
*   Detail performance requirements.

## The PRP Workflow

### How /generate-prp Works

The command operates as follows:

1.  **Research Phase**
    *   Analyzes your codebase for patterns
    *   Searches for similar implementations
    *   Identifies conventions to follow
2.  **Documentation Gathering**
    *   Fetches relevant API docs
    *   Includes library documentation
    *   Adds gotchas and quirks
3.  **Blueprint Creation**
    *   Creates step-by-step implementation plan
    *   Includes validation gates
    *   Adds test requirements
4.  **Quality Check**
    *   Scores confidence level (1-10)
    *   Ensures all context is included

### How /execute-prp Works

1.  **Load Context**: Reads the entire PRP
2.  **Plan**: Creates detailed task list using TodoWrite
3.  **Execute**: Implements each component
4.  **Validate**: Runs tests and linting
5.  **Iterate**: Fixes any issues found
6.  **Complete**: Ensures all requirements met

See `PRPs/EXAMPLE_multi_agent_prp.md` for an example.

## Using Examples Effectively

The `/examples/` folder is **critical** for success. Well-structured examples help the AI assistant.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   How you organize modules
    *   Import conventions
    *   Class/function patterns
2.  **Testing Patterns**
    *   Test file structure
    *   Mocking approaches
    *   Assertion styles
3.  **Integration Patterns**
    *   API client implementations
    *   Database connections
    *   Authentication flows
4.  **CLI Patterns**
    *   Argument parsing
    *   Output formatting
    *   Error handling

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
*   Do not assume the AI understands your preferences.
*   Include specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples
*   More examples lead to better implementations.
*   Show what to do AND what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs include test commands that must pass.
*   The AI will iterate until validations succeed.
*   This ensures working code.

### 4. Leverage Documentation
*   Include official API docs.
*   Add MCP server resources.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add your conventions.
*   Include project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)