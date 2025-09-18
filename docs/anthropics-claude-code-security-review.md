# Secure Your Code with AI: Claude Code Security Reviewer

**Automatically identify and remediate security vulnerabilities in your code with the power of Anthropic's Claude AI, integrated seamlessly into your GitHub workflow.** ([Original Repo](https://github.com/anthropics/claude-code-security-review))

## Key Features

*   **AI-Powered Security Analysis:** Leverages Claude's advanced reasoning to detect vulnerabilities with deep semantic understanding.
*   **Diff-Aware Scanning:** Focuses on changed files in pull requests for efficient analysis.
*   **Automated PR Comments:** Provides detailed security findings directly within your pull requests.
*   **Contextual Understanding:** Analyzes code semantics, not just patterns, for accurate results.
*   **Language Agnostic:** Works with code written in any programming language.
*   **Reduced Noise:** Advanced false positive filtering to highlight critical vulnerabilities.

## Getting Started

Integrate this action into your `.github/workflows/security.yml` file. See the [Quick Start](#quick-start) section in the original README for details.

## Configuration Options

Customize the behavior of the security review with these input parameters:

*   `claude-api-key`: Your Anthropic Claude API key (required).
*   `comment-pr`:  Enable or disable PR comments (default: `true`).
*   `upload-results`:  Upload results as artifacts (default: `true`).
*   `exclude-directories`: Exclude specified directories from scanning (comma-separated).
*   `claude-model`:  Specify the Claude model (defaults to `claude-opus-4-1-20250805`).
*   `claudecode-timeout`: Set the analysis timeout in minutes (default: `20`).
*   `run-every-commit`: Enable analysis on every commit (skips cache check - may increase false positives).
*   `false-positive-filtering-instructions`: Path to custom false positive filtering instructions.
*   `custom-security-scan-instructions`: Path to custom security scan instructions.

### Action Outputs

The action provides the following outputs:

*   `findings-count`: The total number of security findings.
*   `results-file`: The path to the results JSON file.

## How It Works

1.  **PR Analysis:** Analyzes the pull request diff.
2.  **Contextual Review:** Examines code changes in context.
3.  **Finding Generation:** Identifies security issues with details.
4.  **False Positive Filtering:** Reduces noise by removing low-impact findings.
5.  **PR Comments:**  Posts findings as comments.

## Security Analysis Capabilities

### Vulnerabilities Detected

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Issues
*   Data Exposure Risks
*   Cryptographic Weaknesses
*   Input Validation Errors
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Vulnerabilities
*   Code Execution Risks
*   Cross-Site Scripting (XSS)

### False Positive Filtering

The tool filters out low-impact findings like:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Benefits Over Traditional SAST

*   **Contextual Understanding:** Analyzes code semantics and intent for more accurate results.
*   **Reduced False Positives:** AI-powered analysis minimizes noise.
*   **Detailed Explanations:** Provides clear vulnerability explanations and remediation advice.
*   **Customizable:** Adaptable to your organization's specific security needs.

## Integration with Claude Code's `/security-review` Command

The action works alongside the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) in Claude Code. To use it, simply run `/security-review` to perform a comprehensive security review of all pending changes.

Customize the command by copying the [`security-review.md`](https://github.com/anthropics/claude-code-security-review/blob/main/.claude/commands/security-review.md?plain=1) file to your project's `.claude/commands/` folder and editing it.

## Further Customization

You can also configure custom scanning and false positive filtering instructions. See the [`docs/`](docs/) folder for more details.

## Testing

Run tests with:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging.

## License

MIT License - see [LICENSE](LICENSE) file.