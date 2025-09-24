# OWASP Nettacker: Automate Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, Python-based framework empowering cybersecurity professionals and ethical hackers to efficiently assess and secure networks and applications.**  ([Original Repo](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"> <img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

## Key Features

*   **Modular Architecture:** Enables easy customization and control through individual modules for tasks such as port scanning, subdomain enumeration, and vulnerability checks.
*   **Multi-Protocol & Multithreaded Scanning:** Supports a wide range of protocols including HTTP/HTTPS, FTP, SSH, and more, with parallel scanning for enhanced speed.
*   **Comprehensive Reporting:** Generates reports in various formats (HTML, JSON, CSV, TXT) for in-depth analysis.
*   **Built-in Database & Drift Detection:** Stores scan results and allows for comparing results over time to identify changes in infrastructure.
*   **CLI, REST API & Web UI:** Offers flexible control through command-line interface, REST API, and a user-friendly web interface.
*   **Evasion Techniques:** Features configurable delays, proxy support, and randomized user-agents to reduce detection by security systems.
*   **Flexible Target Specifications:** Accepts single IPs, IP ranges, CIDR blocks, domain names, and URLs, supporting target lists.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, and vulnerability scanning to streamline testing.
*   **Recon & Vulnerability Assessment:** Discover live hosts, open ports, services, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly identify exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate common reconnaissance tasks to speed up target identification.
*   **Network Vulnerability Scanning:** Efficiently scan IP ranges and networks using a modular, multithreaded approach.
*   **Shadow IT & Asset Discovery:** Track down forgotten hosts and services.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines for infrastructure change tracking and vulnerability detection.

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

*   Access the Web UI via (https://localhost:5000) or your configured address.
*   The local database is `.nettacker/data/nettacker.db` (SQLite).
*   Default results path is `.nettacker/data/results`.
*   `docker-compose` shares your Nettacker data, so data persists.
*   Find the API key using `docker logs nettacker_nettacker`.
*   More details and install without Docker: [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)

## Important

***THIS SOFTWARE WAS CREATED FOR AUTOMATED PENETRATION TESTING AND INFORMATION GATHERING. YOU MUST USE THIS SOFTWARE IN A RESPONSIBLE AND ETHICAL MANNER. DO NOT TARGET SYSTEMS OR APPLICATIONS WITHOUT OBTAINING PERMISSIONS OR CONSENT FROM THE SYSTEM OWNERS OR ADMINISTRATORS. CONTRIBUTORS WILL NOT BE RESPONSIBLE FOR ANY ILLEGAL USAGE.***

## Links

*   **OWASP Nettacker Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on [https://owasp.slack.com](https://owasp.slack.com)
*   **Donate:** [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)

## Contributions

OWASP Nettacker thrives on community contributions.  Thank you to all the contributors!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We appreciate the organizations and individuals that rely on OWASP Nettacker.
Add your info by submitting a pull request to the [ADOPTERS.md](ADOPTERS.md) file or reach out through GitHub issues!

See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code

OWASP Nettacker has participated in the Google Summer of Code initiative.

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)