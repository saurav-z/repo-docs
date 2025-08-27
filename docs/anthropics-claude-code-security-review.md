# Enhance Your Code Security with AI-Powered Reviews

**Automatically identify and address security vulnerabilities in your code with the Anthropic Claude Code Security Reviewer, a cutting-edge GitHub Action.**  ([Original Repo](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Analysis:** Leverages Anthropic's Claude to provide deep, semantic understanding for vulnerability detection.
*   **Diff-Aware Scanning:** Focuses on code changes within pull requests, optimizing analysis time.
*   **Automated PR Comments:** Comments directly on pull requests, highlighting security findings.
*   **Contextual Understanding:** Analyzes code semantics, moving beyond simple pattern matching.
*   **Language Agnostic:** Supports all programming languages.
*   **Reduced Noise:** Advanced filtering minimizes false positives, focusing on critical vulnerabilities.

## Get Started: Integrate with GitHub Actions

Integrate this GitHub Action to automatically analyze your code with security best practices and reduce the risk of security vulnerabilities:

```yaml
name: Security Review

permissions:
  pull-requests: write  # Needed for leaving PR comments
  contents: read

on:
  pull_request:

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event.pull_request.head.sha || github.sha }}
          fetch-depth: 2
      
      - uses: anthropics/claude-code-security-review@main
        with:
          comment-pr: true
          claude-api-key: ${{ secrets.CLAUDE_API_KEY }}
```

## Configuration Options

Customize the action with these inputs:

*   **`claude-api-key`**: (Required) Your Anthropic Claude API key. *Important: Ensure your API key is enabled for both the Claude API and Claude Code.*
*   **`comment-pr`**: (Default: `true`)  Comment on PRs with findings.
*   **`upload-results`**: (Default: `true`) Upload results as artifacts.
*   **`exclude-directories`**: Comma-separated list of directories to exclude from scanning.
*   **`claude-model`**: (Default: `claude-opus-4-1-20250805`) Claude model name to use. See [Anthropic's model overview](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) for options.
*   **`claudecode-timeout`**: Timeout for ClaudeCode analysis in minutes (default 20).
*   **`run-every-commit`**: Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.
*   **`false-positive-filtering-instructions`**: Path to custom false positive filtering instructions text file.
*   **`custom-security-scan-instructions`**: Path to custom security scan instructions text file to append to audit prompt.

### Action Outputs

*   **`findings-count`**: Total number of security findings.
*   **`results-file`**: Path to the results JSON file.

## How it Works

### Architecture

The core components of the action include:

*   `github_action_audit.py`: Main audit script for GitHub Actions.
*   `prompts.py`: Security audit prompt templates.
*   `findings_filter.py`: False positive filtering logic.
*   `claude_api_client.py`: Claude API client for false positive filtering.
*   `json_parser.py`: Robust JSON parsing utilities.
*   `requirements.txt`: Python dependencies.
*   `test_*.py`: Test suites.
*   `evals/`: Eval tooling to test CC on arbitrary PRs.

### Workflow

1.  **PR Analysis**: Analyzes the pull request diff.
2.  **Contextual Review**: Examines changes with semantic understanding.
3.  **Finding Generation**: Identifies security issues with explanations, severity, and remediation guidance.
4.  **False Positive Filtering**: Reduces noise through advanced filtering.
5.  **PR Comments**: Posts findings directly to the PR.

## Security Analysis Capabilities

### Types of Vulnerabilities Detected

*   **Injection Attacks**: SQL, command, LDAP, XPath, NoSQL, XXE injection
*   **Authentication & Authorization**: Broken authentication, privilege escalation, insecure object references, bypass logic, session flaws
*   **Data Exposure**: Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   **Cryptographic Issues**: Weak algorithms, improper key management, insecure random number generation
*   **Input Validation**: Missing validation, improper sanitization, buffer overflows
*   **Business Logic Flaws**: Race conditions, TOCTOU issues
*   **Configuration Security**: Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain**: Vulnerable dependencies, typosquatting risks
*   **Code Execution**: RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS)**: Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically filters out common, less impactful vulnerabilities:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

Customize the filtering for your needs.

### Benefits Over Traditional SAST

*   **Contextual Understanding**: Comprehends code semantics and intent.
*   **Lower False Positives**: AI-powered analysis reduces noise.
*   **Detailed Explanations**: Provides clear explanations and fixes.
*   **Adaptive Learning**: Customizable with project-specific requirements.

## Installation & Setup

Follow the Quick Start guide above to integrate this action in your GitHub repository.

### Local Development

Use the evaluation framework documentation ([`claudecode/evals/README.md`](claudecode/evals/README.md)) to run the security scanner locally against a specific pull request.

## Claude Code Integration: `/security-review` Command 

This GitHub action integrates with a `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) that provides the same security analysis capabilities as the GitHub Action workflow, but integrated directly into your Claude Code development environment. To use this, simply run `/security-review` to perform a comprehensive security review of all pending changes.

### Customizing the Command

You can customize the `/security-review` command by copying the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder. Edit `security-review.md` to add organization-specific directives to the false positive filtering instructions. 

## Custom Scanning Configuration

To configure custom scanning and false positive filtering instructions, refer to the [`docs/`](docs/) folder.

## Testing

To validate the functionality, run the test suite:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.