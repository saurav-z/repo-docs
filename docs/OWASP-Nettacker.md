# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is a powerful, open-source framework designed for cybersecurity professionals and ethical hackers to efficiently perform reconnaissance, vulnerability assessments, and network audits.**

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![Repo Size](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"> <img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Important Disclaimer:** *This software is intended for authorized penetration testing and information gathering.  Use it responsibly and ethically; do not target systems without explicit permission. The contributors are not responsible for any illegal use.*

![Nettacker Demo GIF](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Perform specific tasks (port scanning, vulnerability checks, credential brute-forcing, etc.) via individual modules, giving you granular control.
*   **Multi-Protocol & Multithreaded Scanning:**  Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Integrated Database & Drift Detection:**  Track scan history to detect new hosts, open ports, and vulnerabilities, ideal for CI/CD pipelines.
*   **CLI, REST API & Web UI:**  Choose from command-line, API, and web interfaces for diverse use cases and integrations.
*   **Evasion Techniques:** Utilize configurable delays, proxy support, and randomized user-agents to evade detection.
*   **Flexible Target Specification:**  Accepts single IPs, IP ranges, CIDR blocks, domain names, and URLs, or load targets from a file.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, misconfiguration checks, and vulnerability scanning to streamline penetration tests.
*   **Reconnaissance & Vulnerability Assessment:** Map hosts, discover open ports, identify services, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly identify exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Reconnaissance:** Automate and scale common reconnaissance tasks like subdomain enumeration to find targets.
*   **Network Vulnerability Scanning:** Efficiently scan IP ranges, CIDR blocks, or subdomains using a modular approach.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged hosts, open ports, and subdomains using scan history and drift detection.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines to track infrastructure changes and detect new vulnerabilities.

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

*   Access the Web GUI using the API Key displayed in the CLI (accessible via  `docker logs nettacker_nettacker`).
*   Web GUI available at `https://localhost:5000` or  `https://nettacker-api.z3r0d4y.com:5000/`.
*   Local database: `.nettacker/data/nettacker.db` (SQLite).
*   Default results path: `.nettacker/data/results`.
*   `docker-compose` shares the nettacker folder.

## Resources

*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **OWASP Nettacker Project Home Page:**  [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Docker Image:**  [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **Installation Guide:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage Guide:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Contributors

OWASP Nettacker thrives on community contributions. Thank you to all our awesome contributors!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We are grateful to organizations and individuals who use Nettacker for their security workflows. If you are using Nettacker, please add your details to the [ADOPTERS.md](ADOPTERS.md) file.

## Google Summer of Code (GSoC)

OWASP Nettacker participates in the Google Summer of Code initiative. Thanks to Google and all the students who have contributed!

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)