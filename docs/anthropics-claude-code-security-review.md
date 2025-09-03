# Enhance Your Code Security with AI-Powered Reviews: Claude Code Security Reviewer

**Stop vulnerabilities before they ship!** The Claude Code Security Reviewer, a GitHub Action, uses Anthropic's Claude Code tool to provide intelligent, context-aware security analysis directly within your pull requests. [View the original repository.](https://github.com/anthropics/claude-code-security-review)

## Key Features:

*   **AI-Powered Analysis:** Leverages Claude's advanced reasoning for deep semantic understanding of your code.
*   **Diff-Aware Scanning:** Focuses analysis on changed files within pull requests, saving time and resources.
*   **Automated PR Comments:** Automatically posts security findings as comments directly in your pull requests.
*   **Contextual Understanding:** Goes beyond simple pattern matching to grasp the intent and meaning of your code.
*   **Language Agnostic:** Works seamlessly with any programming language.
*   **Reduced Noise with Filtering:** Includes built-in filtering to minimize false positives and highlight critical vulnerabilities.

## Get Started Quickly:

Integrate the security review into your `.github/workflows/security.yml` file:

```yaml
name: Security Review

permissions:
  pull-requests: write
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

## Configuration Options:

### Action Inputs

| Input                       | Description                                                                                                                                | Default                   | Required |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------- | -------- |
| `claude-api-key`            | Your Anthropic Claude API key.  *Note*: This key needs to be enabled for both the Claude API and Claude Code usage.                         | None                      | Yes      |
| `comment-pr`                | Whether to comment on PRs with findings                                                                                                    | `true`                    | No       |
| `upload-results`            | Whether to upload results as artifacts                                                                                                     | `true`                    | No       |
| `exclude-directories`       | Comma-separated list of directories to exclude from scanning                                                                              | None                      | No       |
| `claude-model`              | Claude [model name](https://docs.anthropic.com/en/docs/about-claude/models/overview#model-names) to use. Defaults to Opus 4.1.              | `claude-opus-4-1-20250805` | No       |
| `claudecode-timeout`        | Timeout for ClaudeCode analysis in minutes                                                                                                 | `20`                      | No       |
| `run-every-commit`          | Run ClaudeCode on every commit (skips cache check). Warning: May increase false positives on PRs with many commits.                         | `false`                   | No       |
| `false-positive-filtering-instructions` | Path to custom false positive filtering instructions text file                                                                 | None                      | No       |
| `custom-security-scan-instructions` | Path to custom security scan instructions text file to append to audit prompt                                                            | None                      | No       |

### Action Outputs

| Output          | Description                          |
| --------------- | ------------------------------------ |
| `findings-count` | Total number of security findings |
| `results-file`    | Path to the results JSON file       |

## How it Works:

1.  **PR Analysis:** When a pull request is opened, Claude analyzes the diff to understand what changed.
2.  **Contextual Review:** Claude examines the code changes, understanding the purpose and potential security implications.
3.  **Finding Generation:** Security issues are identified with detailed explanations, severity ratings, and remediation guidance.
4.  **False Positive Filtering:** Advanced filtering removes low-impact or false-positive prone findings to reduce noise.
5.  **PR Comments:** Findings are posted as review comments on the specific lines of code.

## Security Analysis Capabilities:

### Types of Vulnerabilities Detected:

*   Injection Attacks (SQL, Command, LDAP, XPath, NoSQL, XXE)
*   Authentication & Authorization Flaws
*   Data Exposure Risks
*   Cryptographic Issues
*   Input Validation Problems
*   Business Logic Flaws
*   Configuration Security Issues
*   Supply Chain Vulnerabilities
*   Code Execution Risks
*   Cross-Site Scripting (XSS)

### False Positive Filtering:

The tool automatically filters out common, low-impact vulnerabilities, including:

*   Denial of Service vulnerabilities
*   Rate limiting concerns
*   Memory/CPU exhaustion issues
*   Generic input validation without proven impact
*   Open redirect vulnerabilities

You can also tune the filtering to match your project's specific security goals.

### Benefits Over Traditional SAST:

*   **Contextual Understanding:** Grasps code meaning, not just patterns.
*   **Lower False Positives:** AI-powered analysis reduces noise by understanding when code is actually vulnerable.
*   **Detailed Explanations:** Provides clear explanations and fixes.
*   **Adaptive Learning:** Customizable with project-specific security needs.

## Installation & Setup:

Follow the Quick Start guide to install the GitHub Action.

### Local Development:

For local testing and development, refer to the [evaluation framework documentation](claudecode/evals/README.md).

## Claude Code Integration: `/security-review` Command

The Claude Code tool ships with a `/security-review` [slash command](https://docs.anthropic.com/en/docs/claude-code/slash-commands) for comprehensive security analysis, integrated directly into your development environment. Run `/security-review` to scan all pending changes.

### Customizing the Command

Customize the command by:

1.  Copying `security-review.md` from this repository to your `.claude/commands/` folder.
2.  Edit the `security-review.md` file to customize the security analysis.

## Custom Scanning Configuration

Configure custom scanning and false positive filtering instructions, in the [`docs/`](docs/) folder for more details.

## Testing:

Run the test suite with:

```bash
cd claude-code-security-review
# Run all tests
pytest claudecode -v
```

## Support

*   Open an issue in this repository.
*   Check the [GitHub Actions logs](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/viewing-workflow-run-history) for debugging information.

## License

MIT License - see the [LICENSE](LICENSE) file for details.