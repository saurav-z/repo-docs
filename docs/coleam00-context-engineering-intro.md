# Context Engineering Template: Build AI-Powered Features Faster

**Tired of struggling with prompt engineering? This template uses context engineering to give your AI coding assistant the information it needs, enabling faster, more reliable feature implementation.** Learn more about this revolutionary approach on the [original repository](https://github.com/coleam00/context-engineering-intro).

## Key Features:

*   **Comprehensive Context:** Provide your AI with everything it needs to succeed, including documentation, examples, and project rules.
*   **Simplified Workflow:** Easily generate detailed implementation plans (PRPs) and execute them to build features.
*   **Reduced Errors:** Minimize AI failures by providing the necessary context.
*   **Consistent Results:** Ensure your AI follows project patterns and conventions.
*   **Self-Correcting:** Leverage validation loops for AI to fix its own mistakes.

## Getting Started: Quick Guide

Follow these steps to get up and running with Context Engineering:

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```

2.  **Customize Project Rules (Optional):**
    *   Edit `CLAUDE.md` to define project-specific guidelines, coding standards, and documentation requirements.

3.  **Add Code Examples (Highly Recommended):**
    *   Place relevant code examples in the `examples/` folder to show the AI specific patterns.

4.  **Create a Feature Request:**
    *   Edit `INITIAL.md` to clearly describe the feature you want to build, including requirements, examples, and documentation links.

5.  **Generate a PRP (Product Requirements Prompt):**
    *   Use the `/generate-prp INITIAL.md` command within Claude Code to generate a detailed implementation plan.

6.  **Execute the PRP:**
    *   Use the `/execute-prp PRPs/your-feature-name.md` command in Claude Code to have the AI implement the feature based on the PRP.

## Table of Contents

*   [What is Context Engineering?](#what-is-context-engineering)
*   [Template Structure](#template-structure)
*   [Step-by-Step Guide](#step-by-step-guide)
*   [Writing Effective INITIAL.md Files](#writing-effective-initialmd-files)
*   [The PRP Workflow](#the-prp-workflow)
*   [Using Examples Effectively](#using-examples-effectively)
*   [Best Practices](#best-practices)

## What is Context Engineering?

Context Engineering is a strategic approach to empower AI coding assistants to build complex software features. Unlike prompt engineering which relies on clever phrasing, context engineering supplies complete context including documentation, examples, and rules.

### Prompt Engineering vs. Context Engineering

**Prompt Engineering:**
*   Focuses on the wording of a prompt.
*   Limited in what the AI can achieve.

**Context Engineering:**
*   Provides a comprehensive system for AI to work with.
*   Allows for robust and multi-step implementations.

### Why Context Engineering Matters

1.  **Reduces AI Failures:** Addresses the most common causes of agent errors.
2.  **Ensures Consistency:** Makes sure your AI follows project patterns.
3.  **Enables Complex Features:** Allows AI to handle multi-step implementations.
4.  **Self-Correcting:** Includes validation steps to enable AI to fix any problems.

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

This template focuses on context engineering for code generation; future updates will introduce RAG and specialized tools.

## Step-by-Step Guide

### 1. Setting Up Global Rules (CLAUDE.md)

The `CLAUDE.md` file defines project-wide rules for your AI assistant. The template includes:

*   **Project awareness**: Reading planning docs, checking tasks
*   **Code structure**: File size limits, module organization
*   **Testing requirements**: Unit test patterns, coverage expectations
*   **Style conventions**: Language preferences, formatting rules
*   **Documentation standards**: Docstring formats, commenting practices

**Customize this template for your project.**

### 2. Create Your Initial Feature Request

Edit `INITIAL.md` to specify your feature request. This should be clear, detailed, and include:

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

**See `INITIAL_EXAMPLE.md` for a complete example of a feature request.**

### 3. Generate the PRP

PRPs (Product Requirements Prompts) act as comprehensive blueprints for implementation, including:

*   Complete context and documentation
*   Implementation steps with validation
*   Error handling patterns
*   Test requirements

Run in Claude Code:
```bash
/generate-prp INITIAL.md
```

**Note:** The slash commands are custom commands defined in `.claude/commands/`. You can view their implementation:
- `.claude/commands/generate-prp.md` - See how it researches and creates PRPs
- `.claude/commands/execute-prp.md` - See how it implements features from PRPs

This command will:

1.  Read your feature request.
2.  Research the codebase and identify patterns.
3.  Search for relevant documentation.
4.  Create a comprehensive PRP in `PRPs/your-feature-name.md`.

### 4. Execute the PRP

After generating the PRP, execute it to implement your feature:

```bash
/execute-prp PRPs/your-feature-name.md
```

The AI assistant will:

1.  Read the complete PRP context.
2.  Create a detailed implementation plan.
3.  Execute each step, validating results.
4.  Run tests and resolve issues.
5.  Verify that all success criteria are met.

## Writing Effective INITIAL.md Files

### Key Sections Explained

**FEATURE**: Provide a specific and detailed description.
*   ❌ "Build a web scraper"
*   ✅ "Build an async web scraper using BeautifulSoup that extracts product data from e-commerce sites, handles rate limiting, and stores results in PostgreSQL"

**EXAMPLES**: Thoroughly utilize the `examples/` folder.
*   Place relevant code patterns in `examples/`.
*   Reference files and patterns.
*   Clarify what aspects of the example the AI should mimic.

**DOCUMENTATION**: Include all relevant resources.
*   API documentation URLs
*   Library guides
*   Server documentation
*   Database schemas

**OTHER CONSIDERATIONS**: Capture critical details.
*   Authentication requirements
*   Rate limits or quotas
*   Common pitfalls
*   Performance requirements

## The PRP Workflow

### How `/generate-prp` Works

The process is designed as follows:

1.  **Research Phase**
    *   Analyzes your code for patterns.
    *   Searches for similar implementations.
    *   Identifies conventions to follow.

2.  **Documentation Gathering**
    *   Fetches API documentation.
    *   Includes library documentation.
    *   Adds gotchas and quirks.

3.  **Blueprint Creation**
    *   Creates step-by-step implementation plan.
    *   Includes validation gates.
    *   Adds test requirements.

4.  **Quality Check**
    *   Scores confidence level (1-10).
    *   Ensures all context is included.

### How `/execute-prp` Works

1.  **Load Context**: Reads the complete PRP.
2.  **Plan**: Creates a detailed task list.
3.  **Execute**: Implements each component.
4.  **Validate**: Runs tests and linting.
5.  **Iterate**: Fixes any issues found.
6.  **Complete**: Ensures all requirements are met.

See `PRPs/EXAMPLE_multi_agent_prp.md` for a detailed example of a generated PRP.

## Using Examples Effectively

The `examples/` directory is **essential** for success. Provide a wealth of examples.

### What to Include in Examples

1.  **Code Structure Patterns**
    *   Module organization
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

### 1. Be Specific in INITIAL.md
*   Assume the AI has no prior knowledge.
*   Define specific requirements and constraints.
*   Reference examples liberally.

### 2. Provide Comprehensive Examples
*   More examples mean better implementations.
*   Show both what to do *and* what not to do.
*   Include error handling patterns.

### 3. Use Validation Gates
*   PRPs have test commands that *must* pass.
*   The AI iterates until validations are successful.
*   This ensures working code on the first attempt.

### 4. Leverage Documentation
*   Include official API docs.
*   Add any relevant server documentation.
*   Reference specific documentation sections.

### 5. Customize CLAUDE.md
*   Add your specific conventions.
*   Include project-specific rules.
*   Define clear coding standards.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)