# Enhance Your Code Security with AI-Powered Reviews: Claude Code Security Reviewer

**Tired of manual security audits?** The Claude Code Security Reviewer GitHub Action leverages Anthropic's Claude AI to automate code security analysis, providing intelligent, context-aware vulnerability detection directly within your pull requests. [Learn more at the original repo](https://github.com/anthropics/claude-code-security-review).

## Key Features:

*   **AI-Driven Analysis:** Utilizes Claude's advanced reasoning capabilities for deep semantic understanding of your code.
*   **Diff-Aware Scanning:** Focuses on changes within pull requests, optimizing scan efficiency.
*   **Automated PR Comments:**  Posts security findings as comments directly in your pull requests for immediate visibility.
*   **Contextual Understanding:**  Goes beyond simple pattern matching, interpreting code's purpose and intent.
*   **Language Agnostic:**  Works seamlessly with any programming language.
*   **False Positive Reduction:** Advanced filtering minimizes noise and prioritizes critical vulnerabilities.

## How It Works

1.  **PR Analysis:** When a pull request is created, Claude analyzes the code changes (diff).
2.  **Contextual Review:** Claude examines the code changes in their context, evaluating potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:**  Filters out common false positives to reduce noise.
5.  **PR Comments:** Findings are presented as review comments on the specific lines of code.

## Installation & Quick Start

Integrate the action into your repository's `.github/workflows/security.yml` file:

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

*   **`claude-api-key`**:  Your Anthropic Claude API key (required). This needs to be enabled for both the Claude API and Claude Code usage.
*   **`comment-pr`**:  Enables or disables PR commenting (defaults to `true`).
*   **Other Configuration Options:**  Customize behavior using additional inputs like `exclude-directories`, `claude-model`, and more.  See the original README for detailed options.

## Security Analysis Capabilities

*   **Vulnerability Detection:** Identifies a wide range of security threats, including:
    *   Injection Attacks (SQLi, Command Injection, etc.)
    *   Authentication and Authorization Flaws
    *   Data Exposure Risks (Hardcoded Secrets, PII Handling)
    *   Cryptographic Vulnerabilities
    *   Input Validation Issues
    *   Business Logic Errors
    *   Configuration Security Weaknesses
    *   Supply Chain Risks
    *   Code Execution vulnerabilities (RCE)
    *   Cross-Site Scripting (XSS)

*   **False Positive Filtering:** The tool automatically excludes findings like:
    *   Denial of Service vulnerabilities
    *   Rate limiting concerns
    *   Memory/CPU exhaustion issues
    *   Generic input validation without proven impact
    *   Open redirect vulnerabilities

## Claude Code Integration: /security-review Command

The action includes a `/security-review` slash command (available in your Claude Code development environment) to perform on-demand security reviews.

### Customizing the Command

1.  Copy the `security-review.md` file from the repository to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to customize the security analysis.

## Advanced Configuration

See the [`docs/`](docs/) folder for configuration options for custom scanning and false positive filtering instructions.

## Testing

Run the test suite to validate functionality:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License. See the [LICENSE](LICENSE) file for details.