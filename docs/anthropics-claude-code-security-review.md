# Enhance Your Code Security with AI-Powered Reviews from Claude Code

**Protect your codebase with the intelligence of Anthropic's Claude Code.** This GitHub Action provides automated, context-aware security analysis directly within your pull requests, helping you identify and address vulnerabilities early in the development cycle.  [Explore the original repository](https://github.com/anthropics/claude-code-security-review).

## Key Features

*   🛡️ **AI-Powered Security Analysis:** Leverages Claude's advanced reasoning to detect vulnerabilities with deep semantic understanding of your code.
*   🔍 **Diff-Aware Scanning:** Focuses on changes within pull requests, optimizing analysis time.
*   💬 **Automated PR Comments:** Highlights security findings directly in your pull request with actionable insights.
*   🧠 **Contextual Understanding:** Goes beyond superficial pattern matching to grasp the intent and meaning of your code.
*   🌐 **Language Agnostic:** Compatible with any programming language.
*   ✅ **Advanced Filtering:** Minimizes noise with built-in false positive reduction, focusing on critical issues.

## Getting Started

Integrate this powerful security tool into your workflow effortlessly with the following steps:

1.  **Add to your `.github/workflows/security.yml`:**

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

2.  **Configure API Key:**  Provide your Anthropic Claude API key as a GitHub secret (`CLAUDE_API_KEY`).  Ensure your key is enabled for both the Claude API and Claude Code.

## Configuration Options

Customize the action's behavior to fit your specific needs:

### Inputs

| Input                      | Description                                                                                                                                | Default                     | Required |
| :------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------- | :-------------------------- | :------- |
| `claude-api-key`          | Your Anthropic Claude API key (enabled for both Claude API and Claude Code).                                                              | None                        | Yes      |
| `comment-pr`               | Whether to post findings as comments on the pull request.                                                                                 | `true`                      | No       |
| `upload-results`           | Whether to upload the analysis results as artifacts.                                                                                        | `true`                      | No       |
| `exclude-directories`    | Comma-separated list of directories to exclude from scanning.                                                                          | None                        | No       |
| `claude-model`             | The Claude model to use (see [model names](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names)).                | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`       | Timeout for ClaudeCode analysis in minutes.                                                                                               | `20`                        | No       |
| `run-every-commit`         | Run ClaudeCode on every commit (skips cache check).  *Warning: May increase false positives on PRs with many commits.*                     | `false`                     | No       |
| `false-positive-filtering-instructions` | Path to a custom text file containing instructions for false positive filtering.                                                      | None                        | No       |
| `custom-security-scan-instructions` | Path to a custom text file to append custom instructions to the audit prompt.                                                            | None                        | No       |

### Outputs

| Output          | Description                      |
| :-------------- | :------------------------------- |
| `findings-count` | The total number of security findings. |
| `results-file`   | The path to the results JSON file.  |

## How It Works

1.  **PR Analysis:**  The action analyzes the pull request's diff.
2.  **Contextual Review:** Claude examines code changes within their context.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation advice.
4.  **False Positive Filtering:**  Advanced filtering reduces noise by removing low-impact issues.
5.  **PR Comments:**  Findings are posted as comments directly in the pull request.

## Security Analysis Capabilities

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE, and more.
*   **Authentication & Authorization:** Broken auth, privilege escalation, insecure object references, and session flaws.
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, and PII handling.
*   **Cryptographic Issues:** Weak algorithms, improper key management, and insecure random number generation.
*   **Input Validation:** Missing validation, improper sanitization, and buffer overflows.
*   **Business Logic Flaws:** Race conditions and TOCTOU issues.
*   **Configuration Security:** Insecure defaults, missing security headers, and permissive CORS.
*   **Supply Chain:** Vulnerable dependencies and typosquatting risks.
*   **Code Execution:** RCE via deserialization, pickle injection, and eval injection.
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS.

### False Positive Filtering

The tool intelligently filters out common false positives:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without impact
*   Open redirect vulnerabilities

Customize this further to align with your security objectives.

### Benefits Over Traditional SAST

*   **Contextual Understanding:** Analyzes code meaning, not just patterns.
*   **Reduced False Positives:** AI-powered analysis minimizes noise.
*   **Detailed Explanations:** Clear explanations of vulnerabilities and solutions.
*   **Customizable:** Adaptable to specific security needs.

## Integration: /security-review Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within your Claude Code environment to perform security analysis on your code with a single command.

### Customizing the Command

1.  Copy `security-review.md` from the repository's `.claude/commands/` folder to your project's `.claude/commands/` folder.
2.  Edit `security-review.md` to adjust the security analysis instructions for your needs.

## Advanced Configuration

Configure custom scanning and false positive filtering through your settings. See the [`docs/`](docs/) folder for more details.

## Testing

Ensure proper functionality with the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for troubleshooting information.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file.