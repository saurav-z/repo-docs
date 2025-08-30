# OWASP Nettacker: Automate Your Penetration Testing and Information Gathering 

**OWASP Nettacker is a powerful, open-source framework for automated penetration testing and information gathering, designed to help you identify and assess security vulnerabilities efficiently.**  ([View on GitHub](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200">
<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER:** This software is for ethical use only.  Do not use it against systems without explicit permission.  Contributors are not responsible for illegal use.

## Key Features

*   **Modular Architecture:** Execute specific tasks like port scanning, vulnerability checks, and credential brute-forcing via modular components, allowing for highly-customized scans.
*   **Multi-Protocol & Multithreaded Scanning:**  Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, and XML-RPC, with parallel scanning capabilities for increased speed.
*   **Comprehensive Reporting:**  Generate reports in HTML, JSON, CSV, and plain text formats to easily analyze and share results.
*   **Built-in Database & Drift Detection:** Store scan results in a database for historical analysis, comparison, and detecting changes in the target environment.
*   **CLI, REST API & Web UI:** Offers a flexible choice of command-line, REST API, and user-friendly web interfaces for seamless integration into different workflows.
*   **Evasion Techniques:** Utilize configurable delays, proxy support, and randomized user-agents to help avoid detection by security systems.
*   **Flexible Target Input:**  Accepts single IPv4s, IP ranges, CIDR blocks, domain names, and full HTTP/HTTPS URLs.
*   **Docker Support:** Easy deployment using Docker, allowing for a simple setup to get you started.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, misconfiguration checks, service discovery, and vulnerability scanning.
*   **Recon & Vulnerability Assessment:** Map hosts, identify open ports, services, default credentials, and directories, then use brute-force attacks or fuzzing.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate and scale subdomain enumeration, directory brute-forcing, and default credential checks.
*   **Network Vulnerability Scanning:** Scan IPs, IP ranges, CIDR blocks, or subdomains efficiently.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged hosts, open ports/services, and subdomains.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker to detect infrastructure changes and new vulnerabilities.

## Quick Start with Docker

### CLI (Docker)

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

### Web UI (Docker)

```bash
$ docker-compose up 
```
*   Use the API Key displayed in the CLI to login to the Web GUI
*   Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Links

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **How to use the Dockerfile:** [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate**: [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More**: [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

##  Contributors
OWASP Nettacker is a community-driven project.  Contributions are welcome!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We’re grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

 See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

*   ☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative
*   🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)