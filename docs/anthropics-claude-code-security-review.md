# Enhance Your Code Security with AI: Claude Code Security Reviewer

**Automatically identify and fix security vulnerabilities in your code with the power of Anthropic's Claude AI.** [Learn more](https://github.com/anthropics/claude-code-security-review)

## Key Features:

*   **AI-Powered Security Analysis:** Leverage Claude's advanced reasoning for deep semantic understanding of your code.
*   **Diff-Aware Scanning:** Focus your security analysis on only the changed files within a pull request.
*   **Automated PR Comments:** Receive insightful comments directly on your pull requests highlighting security findings.
*   **Contextual Understanding:** Goes beyond pattern matching, analyzing the meaning of your code.
*   **Language Agnostic:** Supports all programming languages.
*   **Advanced False Positive Filtering:** Reduces noise and focuses on real vulnerabilities.
*   **Slash Command Integration:** Built-in `/security-review` command for the Claude Code development environment.

## Get Started: Integrate with GitHub Actions

Integrate this tool into your CI/CD pipeline for automated security reviews.

1.  **Add to your repository's `.github/workflows/security.yml`:**

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

2.  **Configuration Options:** Customize the action with the following inputs:

    | Input                      | Description                                                                                                                                    | Default                     | Required |
    | -------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------- | -------- |
    | `claude-api-key`           | Your Anthropic Claude API key.  *Note*: This API key requires both the Claude API and Claude Code usage to be enabled.                              | None                        | Yes      |
    | `comment-pr`               | Whether to post comments on the PR.                                                                                                             | `true`                      | No       |
    | `upload-results`           | Whether to upload results as artifacts.                                                                                                         | `true`                      | No       |
    | `exclude-directories`      | A comma-separated list of directories to exclude from the security scan.                                                                    | None                        | No       |
    | `claude-model`             | [Model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.                         | `claude-opus-4-1-20250805` | No       |
    | `claudecode-timeout`       | Timeout for ClaudeCode analysis in minutes.                                                                                                    | `20`                        | No       |
    | `run-every-commit`         | Run ClaudeCode on every commit (skips cache check). *Warning: May increase false positives on PRs with many commits.*                           | `false`                     | No       |
    | `false-positive-filtering-instructions` | Path to a custom false positive filtering instructions text file.                                                              | None                        | No       |
    | `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt.                                                            | None                        | No       |

3.  **Action Outputs:**

    | Output           | Description                       |
    | ---------------- | --------------------------------- |
    | `findings-count` | Total number of security findings |
    | `results-file`   | Path to the results JSON file      |

## How It Works:

1.  **PR Analysis:** Analyzes the diff in a pull request.
2.  **Contextual Review:**  Examines changes, understanding their context and security implications.
3.  **Finding Generation:** Identifies issues with explanations, severity ratings, and guidance.
4.  **False Positive Filtering:** Removes low-impact findings to reduce noise.
5.  **PR Comments:**  Posts findings as review comments on specific lines.

### Architecture Overview:

*   `claudecode/`
    *   `github_action_audit.py`
    *   `prompts.py`
    *   `findings_filter.py`
    *   `claude_api_client.py`
    *   `json_parser.py`
    *   `requirements.txt`
    *   `test_*.py`
    *   `evals/`

## Security Analysis Capabilities:

### Vulnerability Types Detected:

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization (Broken auth, privilege escalation, insecure object references, bypass logic, session flaws)
*   Data Exposure (Hardcoded secrets, sensitive data logging, information disclosure, PII)
*   Cryptographic Issues (Weak algorithms, improper key management, insecure random numbers)
*   Input Validation (Missing validation, improper sanitization, buffer overflows)
*   Business Logic Flaws (Race conditions, TOCTOU)
*   Configuration Security (Insecure defaults, missing headers, permissive CORS)
*   Supply Chain (Vulnerable dependencies, typosquatting)
*   Code Execution (RCE via deserialization, pickle/eval injection)
*   Cross-Site Scripting (XSS)

### False Positive Filtering:

Automatically excludes:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

## Benefits over Traditional SAST:

*   **Contextual Understanding:** Understands code semantics.
*   **Lower False Positives:**  AI-powered for reduced noise.
*   **Detailed Explanations:** Provides clear explanations and fixes.
*   **Adaptive Learning:** Customizable with organization-specific needs.

## Local Development & Testing:

*   See the [evaluation framework documentation](claudecode/evals/README.md) for local testing.
*   Run tests with: `pytest claudecode -v`

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging.

## License

MIT License - see [LICENSE](LICENSE) file.