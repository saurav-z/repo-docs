# OWASP Nettacker: Automated Penetration Testing Framework

**OWASP Nettacker is a powerful, open-source Python-based framework designed to automate penetration testing, reconnaissance, and vulnerability assessments, helping you identify and address security weaknesses efficiently.**

[Visit the original repo on GitHub](https://github.com/OWASP/Nettacker)

**Disclaimer:** *This software is for ethical use only. Do not use it against systems without explicit permission.*

## Key Features

*   **Modular Architecture:** Easily customize scans with independent modules for tasks like port scanning, subdomain enumeration, and vulnerability checks.
*   **Multi-Protocol & Multithreaded Scanning:** Efficiently scan HTTP/HTTPS, FTP, SSH, SMB, SMTP, and more, with parallel processing for speed.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track and compare scan results over time to identify changes and potential vulnerabilities.
*   **CLI, REST API & Web UI:** Use the CLI, REST API or a user-friendly web interface for flexible integration and control.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and randomized user-agents to bypass detection.
*   **Flexible Target Specification:** Target single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments for efficient penetration testing workflows.
*   **Recon & Vulnerability Assessment:** Map hosts, services, and directories and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate reconnaissance tasks to find targets quickly.
*   **Network Vulnerability Scanning:** Perform large-scale network assessments efficiently.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged or forgotten hosts, ports/services, and subdomains.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect new vulnerabilities.

## Quick Start (Docker)

### CLI

```bash
# Basic port scan on a single IP address:
$ docker run owasp/nettacker -i 192.168.0.1 -m port_scan
# Scan the entire Class C network for any devices with port 22 open:
$ docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22
# Scan all subdomains of 'owasp.org' for http/https services and return HTTP status code
$ docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan
# Display Help
$ docker run owasp/nettacker --help
```

### Web UI

```bash
$ docker-compose up
```

*   Access the Web UI at `https://localhost:5000` (or your configured port).
*   Use the API key from the CLI output to log in.
*   The local database is `.nettacker/data/nettacker.db` (SQLite).
*   Results are saved to `.nettacker/data/results`.
*   `docker-compose` shares your Nettacker folder, preserving data after `docker-compose down`.
*   See the API key with `docker logs nettacker_nettacker`.
*   More details and non-Docker installation: [Installation Guide](https://nettacker.readthedocs.io/en/latest/Installation)

## Contributing

OWASP Nettacker is an open-source project driven by the community. Join us!

## Adopters

We appreciate all the organizations and individuals using Nettacker! Add your info to [ADOPTERS.md](ADOPTERS.md) via pull request or GitHub issue.

## Google Summer of Code

OWASP Nettacker participates in the Google Summer of Code initiative. Thanks to Google and all contributing students!

## Community

*   [OWASP Nettacker Project Home Page](https://owasp.org/nettacker)
*   [Documentation](https://nettacker.readthedocs.io)
*   [Slack](https://owasp.slack.com/archives/CQZGG24FQ)
*   [Docker Image](https://hub.docker.com/r/owasp/nettacker)
*   [OpenHub](https://www.openhub.net/p/OWASP-Nettacker)
*   [Donate](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)