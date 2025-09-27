# OWASP Nettacker: Your All-in-One Automated Penetration Testing Framework

**OWASP Nettacker is a powerful, open-source, Python-based framework designed to automate your penetration testing, reconnaissance, and vulnerability assessment needs.** [View the original repository on GitHub](https://github.com/OWASP/Nettacker).

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER:** *This software is designed for ethical security testing. Use it responsibly and with explicit permission from the target systems' owners.*

## Key Features

*   **Modular Architecture:** Easily customize your scans with individual modules for port scanning, vulnerability checks, and more.
*   **Multi-Protocol & Multithreaded:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scanning for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Database & Drift Detection:** Track historical scan data and changes for enhanced security monitoring and CI/CD integration.
*   **CLI, REST API & Web UI:** Choose your preferred interface for defining and managing scans.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and user-agent randomization to avoid detection.
*   **Flexible Target Specification:** Scan single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessment workflows.
*   **Reconnaissance and Vulnerability Assessment:** Discover live hosts, open ports, services, and perform credential brute-forcing.
*   **Attack Surface Mapping:** Quickly identify exposed hosts, ports, subdomains, and services.
*   **Bug Bounty Recon:** Automate reconnaissance for bug bounty programs.
*   **Network Vulnerability Scanning:** Perform efficient, parallel scans of IPs, IP ranges, and CIDR blocks.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets and services.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into CI/CD pipelines for continuous security checks.

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

## Community and Support

*   **Project Home Page:** [https://owasp.org/nettacker](https://owasp.org/nettacker)
*   **Documentation:** [https://nettacker.readthedocs.io](https://nettacker.readthedocs.io)
*   **Slack:** [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   **Installation:** [https://nettacker.readthedocs.io/en/latest/Installation](https://nettacker.readthedocs.io/en/latest/Installation)
*   **Usage:** [https://nettacker.readthedocs.io/en/latest/Usage](https://nettacker.readthedocs.io/en/latest/Usage)
*   **Docker Image:** [https://hub.docker.com/r/owasp/nettacker](https://hub.docker.com/r/owasp/nettacker)
*   **OpenHub:** [https://www.openhub.net/p/OWASP-Nettacker](https://www.openhub.net/p/OWASP-Nettacker)
*   **Donate**: [https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker](https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker)
*   **Read More**: [https://www.secologist.com/open-source-projects](https://www.secologist.com/open-source-projects)

## Thanks to our awesome contributors!

OWASP Nettacker is an open-source project, built on the principles of collaboration and shared knowledge. The vibrant OWASP community contributes to its development, ensuring that the tool remains up-to-date, adaptable, and aligned with the latest security practices. Thanks to all our awesome contributors! 🚀

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

<img alt="" referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=8e922d16-445a-4c63-b4cf-5152fbbaf7fd" />