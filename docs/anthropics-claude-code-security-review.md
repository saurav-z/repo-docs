# Enhance Your Code Security with AI-Powered Security Reviews

**Stop vulnerabilities before they become problems!** This GitHub Action uses Anthropic's Claude Code, an AI-powered security reviewer, to automatically analyze your code for security flaws. [View the original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Powered Analysis:** Leverages Claude's advanced reasoning to detect vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses on changed files within pull requests for efficient review.
*   **Automated PR Comments:** Directly annotates pull requests with security findings, streamlining the review process.
*   **Contextual Understanding:** Goes beyond pattern matching to understand code semantics and intent.
*   **Language Agnostic:** Works with any programming language.
*   **Reduced Noise:** Advanced filtering minimizes false positives, focusing on critical vulnerabilities.

## How it Works:

1.  **PR Analysis:** On pull request creation, the action analyzes the code diff.
2.  **Contextual Review:** Claude examines the code changes in context, identifying potential security issues.
3.  **Finding Generation:** Security vulnerabilities are identified, with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering reduces noise by removing low-impact or false positive-prone findings.
5.  **PR Comments:** Findings are posted as review comments directly on the relevant lines of code.

## Installation & Setup:

Add the following to your repository's `.github/workflows/security.yml` file:

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

**Important:** Ensure you have a valid Anthropic Claude API key and enable Claude Code usage.

## Configuration Options:

Configure the action using these inputs:

*   `claude-api-key` (Required): Your Anthropic Claude API key.
*   `comment-pr` (Optional, Default: `true`): Comment on PRs with findings.
*   `upload-results` (Optional, Default: `true`): Upload results as artifacts.
*   `exclude-directories` (Optional): Comma-separated list of directories to exclude.
*   `claude-model` (Optional, Default: `claude-opus-4-1-20250805`): Claude model name to use.
*   `claudecode-timeout` (Optional, Default: `20`): Timeout for ClaudeCode analysis in minutes.
*   `run-every-commit` (Optional, Default: `false`): Run ClaudeCode on every commit.
*   `false-positive-filtering-instructions` (Optional): Path to custom false positive filtering instructions.
*   `custom-security-scan-instructions` (Optional): Path to custom security scan instructions.

### Action Outputs

*   `findings-count`: Total number of security findings.
*   `results-file`: Path to the results JSON file.

## Security Analysis Capabilities:

This tool identifies a wide range of vulnerabilities, including:

*   Injection Attacks (SQL, Command, LDAP, etc.)
*   Authentication & Authorization Flaws
*   Data Exposure Issues
*   Cryptographic Weaknesses
*   Input Validation Problems
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Risks
*   Code Execution Vulnerabilities
*   Cross-Site Scripting (XSS)

### False Positive Filtering:

The tool automatically filters out common false positives, such as:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Claude Code Integration: /security-review Command

Use the `/security-review` slash command for security analysis within your Claude Code development environment.  Customize the command by copying the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` directory.

## Testing

To run the test suite locally:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see [LICENSE](LICENSE) file for details.