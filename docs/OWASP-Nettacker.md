# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is a powerful, open-source framework for automating penetration testing and information gathering, designed to help security professionals and ethical hackers identify vulnerabilities efficiently.** [Visit the original repository](https://github.com/OWASP/Nettacker)

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** This software is designed for ethical and responsible use ONLY.  Do not target systems without explicit permission.

## Key Features

*   **Modular Architecture:** Execute specific tasks with independent modules, giving you granular control.
*   **Multi-Protocol & Multithreaded Scanning:** Supports various protocols (HTTP/HTTPS, FTP, SSH, etc.) and parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Database & Drift Detection:**  Store and compare scan results to track changes and identify new vulnerabilities.
*   **CLI, REST API & Web UI:** Choose your preferred interface for defining scans and viewing results.
*   **Evasion Techniques:** Utilize delays, proxies, and user-agent randomization to avoid detection.
*   **Flexible Target Input:** Accept single IPs, ranges, CIDR blocks, domains, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments.
*   **Recon & Vulnerability Assessment:** Discover hosts, services, and credentials.
*   **Attack Surface Mapping:** Quickly identify exposed assets and services.
*   **Bug Bounty Recon:** Streamline reconnaissance for bug bounty programs.
*   **Network Vulnerability Scanning:** Efficiently scan large network ranges.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged and forgotten assets.
*   **CI/CD & Compliance Monitoring:** Integrate into pipelines for continuous security checks.

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
* Use the API Key displayed in the CLI to login to the Web GUI
* Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
* The local database is `.nettacker/data/nettacker.db` (sqlite).
* Default results path is `.nettacker/data/results`
* `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
* To see the API key in you can also run `docker logs nettacker_nettacker`.
* More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Contribute

OWASP Nettacker is a community-driven project.  We welcome contributions from all!

### Adopters

We are grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative
🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

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
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More:** [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)