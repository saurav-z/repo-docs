# Enhance Your Code Security with AI-Powered Reviews from Anthropic's Claude Code

**Automatically identify and address security vulnerabilities in your code with the [Claude Code Security Review GitHub Action](https://github.com/anthropics/claude-code-security-review), leveraging the power of AI for comprehensive code analysis.**

## Key Features

*   🛡️ **AI-Powered Security Analysis:** Uses Claude's advanced reasoning to detect vulnerabilities.
*   🔍 **Diff-Aware Scanning:** Analyzes only the changed files in pull requests, saving time and resources.
*   💬 **Automated PR Comments:** Provides clear feedback with comments directly in your pull requests.
*   💡 **Contextual Understanding:** Analyzes code semantics beyond simple pattern matching for accurate results.
*   🌐 **Language Agnostic:** Supports security reviews for any programming language.
*   🚫 **Advanced False Positive Filtering:** Reduces noise, focusing on real security vulnerabilities.

## Getting Started: Integrate into Your GitHub Workflow

1.  **Add the following to your `.github/workflows/security.yml` file:**

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

2.  **Configuration Options:** Customize the behavior of the security review with these inputs:

    | Input                    | Description                                                                                                                                                                                          | Default                    | Required |
    | :----------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------- | :------- |
    | `claude-api-key`         | Your Anthropic Claude API key (must be enabled for both the Claude API and Claude Code).                                                                                                           | None                       | Yes      |
    | `comment-pr`             | Enable or disable PR comment generation.                                                                                                                                                               | `true`                     | No       |
    | `upload-results`         | Upload results as artifacts.                                                                                                                                                                         | `true`                     | No       |
    | `exclude-directories`    | Comma-separated list of directories to exclude from scanning.                                                                                                                                        | None                       | No       |
    | `claude-model`           | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.                                                      | `claude-opus-4-1-20250805` | No       |
    | `claudecode-timeout`     | Timeout for ClaudeCode analysis in minutes.                                                                                                                                                           | `20`                       | No       |
    | `run-every-commit`       | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                                                                                    | `false`                    | No       |
    | `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file | None | No       |
    | `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt | None | No       |

3.  **Action Outputs:**

    | Output          | Description                                  |
    | :-------------- | :------------------------------------------- |
    | `findings-count` | Total number of security findings.           |
    | `results-file`   | Path to the results JSON file.                 |

## How It Works

1.  **PR Analysis:** The action analyzes the pull request diff to understand the changes.
2.  **Contextual Review:** Claude examines the code changes in context, understanding the purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact or false-positive-prone findings to reduce noise.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities

*   **Injection Attacks:** SQL, command, LDAP, XPath, NoSQL, XXE
*   **Authentication & Authorization:** Broken authentication, privilege escalation, insecure direct object references, bypass logic, session flaws
*   **Data Exposure:** Hardcoded secrets, sensitive data logging, information disclosure, PII handling violations
*   **Cryptographic Issues:** Weak algorithms, improper key management, insecure random number generation
*   **Input Validation:** Missing validation, improper sanitization, buffer overflows
*   **Business Logic Flaws:** Race conditions, time-of-check-time-of-use (TOCTOU) issues
*   **Configuration Security:** Insecure defaults, missing security headers, permissive CORS
*   **Supply Chain:** Vulnerable dependencies, typosquatting risks
*   **Code Execution:** RCE via deserialization, pickle injection, eval injection
*   **Cross-Site Scripting (XSS):** Reflected, stored, and DOM-based XSS

### False Positive Filtering

The tool automatically excludes common false positives like:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Claude Code Integration: `/security-review` Command

Use the `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) within your Claude Code development environment for immediate security analysis of your code changes. Customize the command by copying and editing the `security-review.md` file in your project's `.claude/commands/` folder to tailor the analysis to your project's specific needs.

## Custom Scanning Configuration

Customize the security scan by providing custom scanning and false positive filtering instructions. See the [`docs/`](docs/) folder for more details.

## Testing

Run the test suite:

```bash
cd claude-code-security-review
pytest claudecode -v
```

## Support

*   Open an issue in this repository
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information

## License

MIT License - see [LICENSE](LICENSE) file.