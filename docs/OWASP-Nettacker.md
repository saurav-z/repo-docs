# OWASP Nettacker: Automate Your Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, Python-based framework designed to automate penetration testing, vulnerability assessments, and information gathering for ethical hacking and cybersecurity professionals.**

[View the original repository on GitHub](https://github.com/OWASP/Nettacker)

## Key Features

*   **Modular Architecture:** Easily customize your scans with modules for port scanning, vulnerability checks, and more.
*   **Multi-Protocol & Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, and other protocols, with parallel scanning for speed.
*   **Comprehensive Output:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track scan history and detect changes in your infrastructure.
*   **CLI, REST API & Web UI:** Offers multiple interfaces for flexibility and ease of use.
*   **Evasion Techniques:** Utilize proxies, delays, and randomized user agents to avoid detection.
*   **Flexible Targeting:** Target single IPs, ranges, CIDR blocks, domains, and URLs.

## Use Cases

*   **Penetration Testing:** Streamline reconnaissance, service discovery, and vulnerability assessments.
*   **Recon & Vulnerability Assessment:** Map hosts, ports, and services, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly identify exposed hosts, ports, and services for internal and external assets.
*   **Bug Bounty Recon:** Automate common reconnaissance tasks to find targets quickly.
*   **Network Vulnerability Scanning:** Efficiently scan large networks and subdomains.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged or forgotten assets.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect new vulnerabilities.

## Quick Start with Docker

**CLI:**

```bash
# Basic port scan
docker run owasp/nettacker -i 192.168.0.1 -m port_scan

# Scan a network for open port 22
docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22

# Scan subdomain for http/https
docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan

# See all the options
docker run owasp/nettacker --help
```

**Web UI:**

```bash
docker-compose up 
```

*   Access the Web GUI at `https://localhost:5000` (or your Docker host's IP).
*   Use the API Key displayed in the CLI to login.
*   Data is stored in the local database: `.nettacker/data/nettacker.db` (sqlite).
*   Results are saved in `.nettacker/data/results`.

## Important

***Disclaimer:** This software is intended for ethical and authorized use only. Always obtain permission before scanning systems or applications. Contributors are not responsible for misuse.*

## Community and Resources

*   **Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)

## Contributing

OWASP Nettacker is an open-source project and welcomes contributions from the community. 

## Adopters

We are grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker. See the [ADOPTERS.md](ADOPTERS.md) for details. If you are using Nettacker, we’d love to hear from you!

## Google Summer of Code (GSoC)

OWASP Nettacker participated in the Google Summer of Code program. Thanks to Google and all the students!

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)