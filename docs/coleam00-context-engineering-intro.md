# Context Engineering: The Future of AI Coding Assistants

**Unlock the full potential of AI coding with Context Engineering, a comprehensive approach that provides AI assistants with the information they need to deliver high-quality code, every time.** ([View the original repo](https://github.com/coleam00/context-engineering-intro))

## Key Features

*   **Comprehensive Context:** Move beyond simple prompts. This template provides a complete system for supplying your AI assistant with the necessary information to understand your project and requirements.
*   **Reduced AI Failures:** By equipping AI with context, you drastically reduce the chances of errors and unexpected behavior.
*   **Consistent Code:** Enforce project-specific patterns, conventions, and standards for unified code.
*   **Complex Feature Implementation:** Handle intricate, multi-step features with ease by providing step-by-step instructions and validation.
*   **Self-Correcting Capabilities:** Validation loops enable the AI to identify and fix its mistakes, leading to reliable outputs.

## Getting Started

```bash
# 1. Clone the Template
git clone https://github.com/coleam00/Context-Engineering-Intro.git
cd Context-Engineering-Intro

# 2. Set Project Rules (Optional)
# Edit CLAUDE.md for project-specific guidelines (e.g., coding style, testing)

# 3. Add Code Examples (Recommended)
# Place relevant code examples into the examples/ folder

# 4. Create a Feature Request
# Edit INITIAL.md with your desired feature specifications

# 5. Generate a PRP (Product Requirements Prompt)
# In Claude Code, run:
/generate-prp INITIAL.md

# 6. Execute the PRP to Implement the Feature
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

Context Engineering is a superior approach to prompt engineering:

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**

*   Focuses on crafting specific wording.
*   Limited by how you phrase a task.
*   Comparable to giving a sticky note.

**Context Engineering:**

*   A complete system that provides all the context.
*   Includes comprehensive documentation, working examples, and validation methods.
*   Comparable to giving a full screenplay with detailed instructions.

### Why Context Engineering Matters

1.  **Mitigates AI Failures:** Context Engineering is a robust way to prevent agent failures.
2.  **Ensures Code Consistency:** Your AI will follow project-defined patterns and standards.
3.  **Facilitates Complex Features:** Implement multi-step features with appropriate context.
4.  **Self-Correcting**: Validation loops allow AI to fix its own mistakes.

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

This template focuses on providing you with a solid foundation for Context Engineering.

## Step-by-Step Guide

### 1. Define Global Rules (CLAUDE.md)

The `CLAUDE.md` file sets the project-wide guidelines that your AI assistant will use in every interaction. The template includes:

*   **Project Awareness:** Reading planning documents and checking tasks.
*   **Code Structure:** File size limits and module organization.
*   **Testing Requirements:** Unit test patterns and code coverage expectations.
*   **Style Conventions:** Language preferences and formatting rules.
*   **Documentation Standards:** Docstring formats and commenting practices.

**You can either customize the template for your project or use the provided template as-is.**

### 2. Create Your Initial Feature Request

Describe what you want to build in `INITIAL.md`:

```markdown
## FEATURE:
[Describe what you want to build - be specific about functionality and requirements]

## EXAMPLES:
[List any example files in the examples/ folder and explain how they should be used]

## DOCUMENTATION:
[Include links to relevant documentation, APIs, or MCP server resources]

## OTHER CONSIDERATIONS:
[Mention any gotchas, specific requirements, or things AI assistants commonly miss]
```

**See `INITIAL_EXAMPLE.md` for a sample feature request.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) provide comprehensive blueprints for implementing features. They include:

*   Complete context and documentation.
*   Step-by-step implementation with validation gates.
*   Error handling and testing requirements.

They are similar to PRDs (Product Requirements Documents) but tailored for AI coding assistants.

Run in Claude Code:

```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

The `$ARGUMENTS` variable in these commands receives whatever you pass after the command name (e.g., `INITIAL.md` or `PRPs/your-feature.md`).

This command will:

1.  Read your feature request.
2.  Analyze the codebase for existing patterns.
3.  Search for relevant documentation.
4.  Generate a PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

Once a PRP is generated, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI coding assistant will:

1.  Read all context from the PRP.
2.  Develop a detailed implementation plan.
3.  Execute each step, validating the results.
4.  Run tests and fix any encountered issues.
5.  Ensure that all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Section Breakdown

**FEATURE:** Provide a detailed and complete description.

*   **Avoid:** "Build a web scraper."
*   **Use:** "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL."

**EXAMPLES:** Leverage the `examples/` folder.

*   Place relevant code patterns in `examples/`.
*   Refer to specific files and patterns to be followed.
*   Clarify what aspects should be mimicked.

**DOCUMENTATION:** Include all pertinent resources.

*   API documentation URLs.
*   Library guides.
*   MCP server documentation.
*   Database schemas.

**OTHER CONSIDERATIONS:** Capture important nuances.

*   Authentication requirements.
*   Rate limits or quotas.
*   Common mistakes.
*   Performance needs.

## The PRP Workflow

### How /generate-prp Works

The command follows these steps:

1.  **Research Phase:**
    *   Analyzes your codebase to identify patterns.
    *   Searches for related implementations.
    *   Identifies conventions to follow.
2.  **Documentation Gathering:**
    *   Fetches relevant API documentation.
    *   Includes library documentation.
    *   Adds any important limitations.
3.  **Blueprint Creation:**
    *   Constructs a step-by-step implementation plan.
    *   Incorportates validation gates.
    *   Adds testing requirements.
4.  **Quality Assurance:**
    *   Assigns a confidence level (1-10).
    *   Ensures that all context is included.

### How /execute-prp Works

1.  **Load Context:** Reads the entire PRP.
2.  **Plan:** Creates a detailed task list.
3.  **Execute:** Implements each component.
4.  **Validate:** Runs tests and linting.
5.  **Iterate:** Fixes any issues found.
6.  **Complete:** Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a sample.

## Using Examples Effectively

The `examples/` folder is essential. AI coding assistants perform best with existing patterns.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   Module organization.
    *   Import conventions.
    *   Class/function patterns.
2.  **Testing Patterns**
    *   Test file structure.
    *   Mocking approaches.
    *   Assertion styles.
3.  **Integration Patterns**
    *   API client implementations.
    *   Database connections.
    *   Authentication flows.
4.  **CLI Patterns**
    *   Argument parsing.
    *   Output formatting.
    *   Error handling.

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

### 1. Be Specific in INITIAL.md

*   Be explicit about your preferences.
*   Provide specific requirements and constraints.
*   Include plenty of examples.

### 2. Provide Comprehensive Examples

*   More examples result in better implementations.
*   Demonstrate what to do and what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates

*   PRPs use test commands that must pass.
*   The AI will iterate until all tests are successful.
*   This guarantees working code on the first try.

### 4. Leverage Documentation

*   Provide official API documentation.
*   Include MCP server resources.
*   Refer to specific sections of documentation.

### 5. Customize CLAUDE.md

*   Add your coding conventions.
*   Incorporate project-specific rules.
*   Define coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)