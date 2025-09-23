# OWASP Nettacker: Automate Your Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, Python-based penetration testing framework that helps cybersecurity professionals and ethical hackers identify and assess network vulnerabilities efficiently.** ([View the original repository](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"> <img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER: This software is intended for ethical use only.  Do not use Nettacker against systems without explicit permission from the owners.**

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Execute specific tasks independently, providing granular control over your scans.
*   **Multi-Protocol & Multithreaded Scanning:**  Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC and more, with parallel scanning capabilities for speed.
*   **Comprehensive Reporting:** Export results in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:**  Store scan data for historical analysis and identify changes.
*   **CLI, REST API & Web UI:**  Choose your preferred interface for managing and viewing scan results.
*   **Evasion Techniques:**  Utilize proxy support, configurable delays, and randomized user-agents to minimize detection.
*   **Flexible Target Input:**  Accept single IPs, IP ranges, CIDR blocks, domains, and URLs, or load targets from a file.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, vulnerability assessments, and service discovery.
*   **Reconnaissance & Vulnerability Assessment:** Map live hosts, open ports, services, and directories, and perform brute-force testing.
*   **Attack Surface Mapping:** Quickly discover exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate subdomain enumeration, directory brute-forcing, and credential checks.
*   **Network Vulnerability Scanning:**  Assess IPs, IP ranges, CIDR blocks, or subdomains using a modular approach.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets using historical data and drift detection.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker to track infrastructure changes and detect new vulnerabilities.

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

*   Use the API Key displayed in the CLI to log in to the Web GUI.
*   Web GUI accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

##  Community & Support

*   OWASP Nettacker Project Home Page: [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   Documentation: [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   Slack: [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   Installation: [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   Usage: [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   GitHub repo: [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   Docker Image: [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   How to use the Dockerfile: [https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker](https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker)
*   OpenHub: [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate**: [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More**: [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Contributors

OWASP Nettacker thrives thanks to its community. We are grateful to all contributors for their support.

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

##  Adopters

We appreciate all organizations, community projects, and individuals who use OWASP Nettacker. If you use Nettacker, please let us know by adding your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues.

See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative.
🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)