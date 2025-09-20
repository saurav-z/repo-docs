# OWASP Nettacker: Automate Your Penetration Testing and Information Gathering

OWASP Nettacker is a powerful, open-source framework designed to automate penetration testing and information gathering tasks, helping security professionals and ethical hackers identify vulnerabilities effectively.  ([Original Repository](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)


<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">


**DISCLAIMER: This software is designed for ethical and responsible use.  Always obtain permission before conducting any security assessments.**

## Key Features

*   **Modular Architecture:** Easily customize your scans with a modular design, allowing you to select specific tasks like port scanning, vulnerability checks, or credential brute-forcing.
*   **Multi-Protocol & Multithreaded Scanning:** Scan rapidly using multiple protocols (HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC) and multithreading.
*   **Comprehensive Output:** Generate detailed reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track changes and identify new vulnerabilities with a built-in database that stores historical scan data.
*   **CLI, REST API & Web UI:** Choose your preferred interface: command-line, REST API for automation, or user-friendly web interface.
*   **Evasion Techniques:**  Use configurable delays, proxy support, and randomized user-agents to reduce detection.
*   **Flexible Target Support:**  Scan single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, vulnerability assessments, and service discovery.
*   **Recon & Vulnerability Assessment:** Map hosts, ports, services, and perform brute-forcing with ease.
*   **Attack Surface Mapping:** Quickly discover exposed assets, subdomains, and services.
*   **Bug Bounty Recon:** Automate and scale common reconnaissance tasks.
*   **Network Vulnerability Scanning:** Perform efficient network assessments.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets with historical data and drift detection.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines to track infrastructure changes.

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

*   Access the Web GUI at `https://localhost:5000` or `https://nettacker-api.z3r0d4y.com:5000/`.
*   Use the API Key from the CLI output to log in.
*   Database is located at `.nettacker/data/nettacker.db` (SQLite).
*   Results are saved in `.nettacker/data/results`.
*   `docker-compose` shares your nettacker folder.
*   To see the API key, run `docker logs nettacker_nettacker`.
*   For more details and installation without docker, visit the [documentation](https://nettacker.readthedocs.io/en/latest/Installation).

## Contributions and Community

OWASP Nettacker is an open-source project driven by a collaborative community.  We welcome contributions from everyone!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We're grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code

☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative. 🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

## Further Information

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **How to use the Dockerfile:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)