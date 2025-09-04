# Enhance Your Code Security with the Claude Code Security Reviewer

**Identify and fix vulnerabilities effortlessly with the AI-powered Claude Code Security Reviewer, a GitHub Action that brings advanced security analysis directly to your pull requests.** ([View the original repository](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Analysis:** Leverages Anthropic's Claude to understand code semantics and detect vulnerabilities with deep reasoning capabilities.
*   **Diff-Aware Scanning:** Efficiently analyzes only the changed files within your pull requests.
*   **Automated PR Comments:** Automatically posts security findings directly within your pull requests for immediate feedback.
*   **Contextual Understanding:** Goes beyond simple pattern matching to understand the code's purpose and potential security implications.
*   **Language Agnostic:** Works with a wide variety of programming languages.
*   **Advanced False Positive Filtering:** Reduces noise by filtering out low-impact or false-positive prone findings.

## How It Works: A Streamlined Security Workflow

1.  **PR Analysis:** On pull request creation, Claude analyzes the code changes.
2.  **Contextual Review:** The AI examines the code within its context to understand purpose and potential vulnerabilities.
3.  **Finding Generation:** Identifies security issues, providing detailed explanations, severity levels, and remediation guidance.
4.  **False Positive Filtering:** Minimizes noise by filtering low-impact or false-positive findings.
5.  **PR Comments:** Findings are posted as comments directly on the specific lines of code.

## Installation & Setup: Get Started in Minutes

### GitHub Actions

1.  Add the following to your repository's `.github/workflows/security.yml`:

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

2.  **Configuration Options:**

    *   `claude-api-key`: Your Anthropic Claude API key (required).  Ensure the key is enabled for both the Claude API and Claude Code usage.
    *   `comment-pr`:  Whether to post comments on PRs (default: `true`).
    *   `upload-results`:  Whether to upload results as artifacts (default: `true`).
    *   `exclude-directories`:  Comma-separated list of directories to skip.
    *   `claude-model`:  Claude model name (default: `claude-opus-4-1-20250805`). See the [model overview](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) for options.
    *   `claudecode-timeout`:  Timeout for the analysis in minutes (default: `20`).
    *   `run-every-commit`:  Run on every commit (may increase false positives on PRs with many commits) (default: `false`).
    *   `false-positive-filtering-instructions`:  Path to a custom filtering instructions text file.
    *   `custom-security-scan-instructions`:  Path to a custom instructions text file to append to the audit prompt.

    **Outputs:**

    *   `findings-count`: Total number of security findings.
    *   `results-file`: Path to the results JSON file.

### Local Development

To run locally, see the [evaluation framework documentation](claudecode/evals/README.md).

## Security Analysis Capabilities: Comprehensive Vulnerability Coverage

### Types of Vulnerabilities Detected

*   Injection Attacks (SQL, command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Issues (broken auth, privilege escalation, etc.)
*   Data Exposure (hardcoded secrets, sensitive logging)
*   Cryptographic Issues (weak algorithms, key management)
*   Input Validation Problems (missing validation, sanitization)
*   Business Logic Flaws (race conditions, TOCTOU)
*   Configuration Security Issues (insecure defaults, missing headers)
*   Supply Chain Vulnerabilities (vulnerable dependencies, typosquatting)
*   Code Execution (RCE via deserialization, pickle, eval)
*   Cross-Site Scripting (XSS) (reflected, stored, DOM-based)

### Enhanced False Positive Filtering

The tool automatically excludes many common false-positive prone findings, including:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can customize the filtering based on your project's security goals.

### Benefits Over Traditional SAST

*   **Contextual Understanding:**  The AI understands the code's purpose, not just patterns.
*   **Reduced False Positives:**  AI-powered analysis significantly reduces noise.
*   **Detailed Explanations:**  Clear explanations of vulnerabilities and how to fix them are provided.
*   **Customization:**  Adaptable to your organization's specific security needs.

## Claude Code Integration: Slash Command for On-Demand Security Reviews

Use the `/security-review` slash command within your Claude Code development environment for comprehensive security reviews of all pending changes.

### Customizing the Command

1.  Copy the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file from this repository to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis (e.g., add org-specific directions).

## Custom Scanning Configuration

Customize your scanning and false positive filtering instructions.  See the [`docs/`](docs/) folder for details.

## Testing

To validate functionality, run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

For issues or questions:

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.