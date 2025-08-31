# Enhance Your Code Security with AI-Powered Reviews using Claude Code

**Automatically detect security vulnerabilities in your code with the power of Anthropic's Claude, integrating seamlessly into your GitHub workflow.** [Explore the original repo](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   🛡️ **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of your code.
*   🔍 **Diff-Aware Scanning:** Focuses on the changed files within pull requests for efficient analysis.
*   💬 **PR Comments:** Automatically posts security findings directly as comments within your pull requests.
*   🧠 **Contextual Understanding:** Goes beyond basic pattern matching to grasp the underlying semantics of your code.
*   🌐 **Language Agnostic:** Works with a wide range of programming languages.
*   🚫 **False Positive Filtering:** Advanced filtering to reduce noise and highlight the most critical vulnerabilities.

## Quick Start - Integrating with GitHub Actions

To integrate the Claude Code Security Reviewer into your repository, add the following snippet to your `.github/workflows/security.yml` file:

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

### Configuration Options

You can customize the behavior of the action using the following inputs:

*   `claude-api-key`: (Required) Your Anthropic Claude API key, enabled for both the Claude API and Claude Code usage.
*   `comment-pr`: (Optional, default: `true`) Whether to comment on PRs with findings.
*   `upload-results`: (Optional, default: `true`) Whether to upload results as artifacts.
*   `exclude-directories`: (Optional) Comma-separated list of directories to exclude from scanning.
*   `claude-model`: (Optional, default: `claude-opus-4-1-20250805`) Claude model name to use.
*   `claudecode-timeout`: (Optional, default: `20`) Timeout for ClaudeCode analysis in minutes.
*   `run-every-commit`: (Optional, default: `false`) Run ClaudeCode on every commit.
*   `false-positive-filtering-instructions`: (Optional) Path to custom false positive filtering instructions text file.
*   `custom-security-scan-instructions`: (Optional) Path to custom security scan instructions text file to append to audit prompt.

### Action Outputs

*   `findings-count`: The total number of security findings.
*   `results-file`: The path to the results JSON file.

## How It Works

1.  **PR Analysis:** When a pull request is opened, Claude analyzes the diff to understand the code changes.
2.  **Contextual Review:** Claude examines the changes in context, considering their purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact and false positive prone findings.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities

### Vulnerabilities Detected

The Claude Code Security Reviewer can identify a broad range of security vulnerabilities, including:

*   Injection Attacks (SQLi, Command Injection, etc.)
*   Authentication & Authorization Flaws
*   Data Exposure and PII Handling Issues
*   Cryptographic Vulnerabilities
*   Input Validation Errors
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Risks
*   Code Execution Risks (RCE, etc.)
*   Cross-Site Scripting (XSS)

### False Positive Filtering Details

The tool automatically filters a variety of low-impact and false positive prone findings to focus on high-impact vulnerabilities:
* Denial of Service vulnerabilities
* Rate limiting concerns
* Memory/CPU exhaustion issues
* Generic input validation without proven impact
* Open redirect vulnerabilities

The false positive filtering can also be tuned as needed for a given project's security goals.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Understands code semantics and intent, not just patterns
*   **Lower False Positives:** AI-powered analysis reduces noise by understanding when code is actually vulnerable
*   **Detailed Explanations:** Provides clear explanations of why something is a vulnerability and how to fix it
*   **Adaptive Learning:** Can be customized with organization-specific security requirements

## Claude Code Integration: `/security-review` Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) in your Claude Code development environment for a comprehensive security review of your pending changes.

### Customizing the Command

Customize the `/security-review` command by copying and editing the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file in your project's `.claude/commands/` folder.

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions; see the [`docs/`](docs/) folder for details.

## Testing

Run the test suite using:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.