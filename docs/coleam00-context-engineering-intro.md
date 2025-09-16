# Context Engineering Template: Revolutionize AI Coding with Comprehensive Context

**Unlock the power of AI coding assistants by providing them with a complete understanding of your project through Context Engineering.** This template provides a robust framework to build and deploy AI-driven code generation, improving the accuracy and efficiency of your coding workflow.

[View the original repo](https://github.com/coleam00/context-engineering-intro)

## Key Features:

*   **Comprehensive Context:** Go beyond prompt engineering with a complete system encompassing documentation, examples, rules, and validation.
*   **Reduced AI Failures:** Minimize agent failures by providing essential context for accurate code generation.
*   **Consistent Code:** Ensure AI follows your project patterns, conventions, and coding standards.
*   **Complex Feature Implementation:** Empower AI to handle multi-step implementations with properly structured context.
*   **Self-Correcting:** Utilize validation loops that enable AI to identify and resolve errors automatically.

## Getting Started

1.  **Clone the Template:**
    ```bash
    git clone https://github.com/coleam00/Context-Engineering-Intro.git
    cd Context-Engineering-Intro
    ```
2.  **Define Project Rules (Optional):**
    *   Customize project-specific guidelines in `CLAUDE.md`.
3.  **Add Code Examples (Recommended):**
    *   Populate the `examples/` folder with relevant code samples demonstrating patterns.
4.  **Create Feature Requests:**
    *   Edit `INITIAL.md` to specify feature requirements.
5.  **Generate a Product Requirements Prompt (PRP):**
    *   Use the following command within Claude Code:
        ```bash
        /generate-prp INITIAL.md
        ```
6.  **Execute the PRP:**
    *   Run the following command in Claude Code to implement the feature:
        ```bash
        /execute-prp PRPs/your-feature-name.md
        ```

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

## Deep Dive into Context Engineering

### What is Context Engineering?

Context Engineering surpasses traditional prompt engineering by offering a comprehensive approach to providing context to AI coding assistants.

**Prompt Engineering:** Focuses on clever wording and specific phrasing, akin to a sticky note.

**Context Engineering:** Delivers a complete system with documentation, examples, rules, patterns, and validation, similar to writing a full screenplay.

### Why Context Engineering Matters

*   **Minimizes AI Failures**: Addresses context deficiencies, the primary cause of agent failures.
*   **Guarantees Consistency**: Enforces project patterns and coding standards.
*   **Facilitates Complex Features**: Enables AI to manage intricate, multi-step implementations.
*   **Enables Self-Correction**: Implements validation loops, allowing AI to resolve errors.

## Step-by-Step Guide

### 1.  Setting Up Global Rules (CLAUDE.md)

`CLAUDE.md` defines project-wide rules for the AI assistant, encompassing:

*   Project awareness (reading docs, checking tasks)
*   Code structure guidelines (file limits, module organization)
*   Testing requirements (unit test patterns, coverage)
*   Style conventions (language preferences, formatting)
*   Documentation standards (docstring formats, comments)

**Customize the template to align with your project's requirements.**

### 2. Creating Your Initial Feature Request

*   Edit `INITIAL.md` to define your feature requests:

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

*   Refer to `INITIAL_EXAMPLE.md` for a comprehensive example.

### 3. Generating the PRP (/generate-prp)

PRPs (Product Requirements Prompts) are comprehensive implementation blueprints.

*   **Process:**
    1.  Analyzes the codebase for patterns.
    2.  Searches and incorporates relevant documentation.
    3.  Generates a step-by-step implementation plan including validation and test requirements.

*   **Run:**

```bash
/generate-prp INITIAL.md
```

*   The `/generate-prp` command leverages custom commands in `.claude/commands/`.  See the implementation for insights:
    *   `.claude/commands/generate-prp.md`
    *   `.claude/commands/execute-prp.md`

### 4. Executing the PRP (/execute-prp)

Implement your feature with the following command:

```bash
/execute-prp PRPs/your-feature-name.md
```

*   **Execution Steps**:
    1.  Loads the entire PRP context.
    2.  Creates a detailed task list for implementation.
    3.  Executes each step, incorporating validation.
    4.  Runs tests and fixes any issues.
    5.  Confirms the fulfillment of all requirements.

## Writing Effective INITIAL.md Files

### Key Sections

*   **FEATURE:** Be specific and detailed.
*   **EXAMPLES:** Utilize the `examples/` folder extensively.
*   **DOCUMENTATION:** Include all relevant resources (API documentation, guides, etc.).
*   **OTHER CONSIDERATIONS:** Address authentication, rate limits, and common pitfalls.

## The PRP Workflow

### How /generate-prp Works

1.  **Research:** Analyzes the codebase for existing patterns.
2.  **Documentation Gathering:** Fetches relevant API documentation.
3.  **Blueprint Creation:** Produces a step-by-step implementation plan with validation gates and test requirements.
4.  **Quality Check:** Evaluates confidence and confirms context inclusion.

### How /execute-prp Works

1.  Load Context.
2.  Plan.
3.  Execute.
4.  Validate.
5.  Iterate.
6.  Complete.

## Using Examples Effectively

The `examples/` folder is essential for guiding AI coding assistants.

### What to Include

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

1.  **Be Explicit in INITIAL.md** – Define requirements and reference examples.
2.  **Provide Comprehensive Examples** – More examples lead to better implementations.
3.  **Use Validation Gates** – PRPs integrate test commands to ensure functionality.
4.  **Leverage Documentation** – Incorporate API documentation.
5.  **Customize CLAUDE.md** – Define coding standards and project-specific rules.

## Resources

*   [Claude Code Documentation](https://docs.anthropic.com/en/docs/claude-code)
*   [Context Engineering Best Practices](https://www.philschmid.de/context-engineering)