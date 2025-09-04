# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, powerful framework designed to automate penetration testing and reconnaissance tasks, helping security professionals discover vulnerabilities and assess network security efficiently.**

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![Repo Size](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200">
<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** *This software is intended for authorized penetration testing and security assessments. Always obtain explicit permission before using Nettacker on any system or network. Contributors are not responsible for any misuse of the software.*

![Nettacker Demo](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Execute specific tasks (port scanning, vulnerability checks, etc.) via independent modules.
*   **Multi-Protocol & Multithreaded Scanning:** Supports various protocols (HTTP/HTTPS, FTP, SSH, etc.) with parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track scan history and identify changes in your environment.
*   **CLI, REST API & Web UI:** Use the command-line interface, REST API, or a web interface for flexible control.
*   **Evasion Techniques:** Employ delays, proxy support, and user-agent randomization to avoid detection.
*   **Flexible Target Input:** Scan single IPs, ranges, CIDR blocks, domains, or URLs, either individually or from a list.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, service discovery, and vulnerability assessments.
*   **Reconnaissance & Vulnerability Assessment:** Discover live hosts, open ports, services, and perform brute-force attacks.
*   **Attack Surface Mapping:** Quickly identify exposed assets, ideal for internal and external assessments.
*   **Bug Bounty Reconnaissance:** Automate common reconnaissance tasks to accelerate target identification.
*   **Network Vulnerability Scanning:** Scan IPs, IP ranges, and subdomains efficiently at scale.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged hosts, open ports/services, and subdomains over time.
*   **CI/CD & Compliance Monitoring:** Integrate into pipelines to detect changes and vulnerabilities.

## Quick Start (Docker)

### CLI

```bash
# Basic port scan on a single IP address:
docker run owasp/nettacker -i 192.168.0.1 -m port_scan
# Scan the entire Class C network for any devices with port 22 open:
docker run owasp/nettacker -i 192.168.0.0/24 -m port_scan -g 22
# Scan all subdomains of 'owasp.org' for http/https services and return HTTP status code
docker run owasp/nettacker -i owasp.org -d -s -m http_status_scan
# Display Help
docker run owasp/nettacker --help
```

### Web UI

```bash
docker-compose up
```
* Use the API Key displayed in the CLI to login to the Web GUI
* Web GUI is accessible from your (https://localhost:5000) or https://nettacker-api.z3r0d4y.com:5000/ (pointed to your localhost)
* The local database is `.nettacker/data/nettacker.db` (sqlite).
* Default results path is `.nettacker/data/results`
* `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
* To see the API key in you can also run `docker logs nettacker_nettacker`.
* More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Useful Links

*   **GitHub Repository:** [https://github.com/OWASP/Nettacker](https://github.com/OWASP/Nettacker)
*   **Documentation:** https://nettacker.readthedocs.io
*   **OWASP Nettacker Project Home Page:** https://owasp.org/nettacker
*   **Docker Image:** https://hub.docker.com/r/owasp/nettacker
*   **Installation:** https://nettacker.readthedocs.io/en/latest/Installation
*   **Usage:** https://nettacker.readthedocs.io/en/latest/Usage
*   **How to use the Dockerfile:** https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **OpenHub:** https://www.openhub.net/p/OWASP-Nettacker
*   **Donate:** https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker
*   **Read More**: https://www.secologist.com/open-source-projects

## Contributing

OWASP Nettacker thrives on community contributions.  We welcome your help in making it even better.  Review the [ADOPTERS.md](ADOPTERS.md) file to see how you can contribute.

## Contributors

Thanks to all our awesome contributors! 🚀

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We’re grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

 See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

*   ☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative
*   🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers Over Time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)