# OWASP Nettacker: Automated Penetration Testing and Information Gathering

**OWASP Nettacker is an open-source, Python-based framework designed to automate security assessments, helping you identify vulnerabilities and strengthen your network defenses.** [View on GitHub](https://github.com/OWASP/Nettacker)

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)


<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"> <img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**Disclaimer:** *This software is intended for ethical and responsible use in penetration testing and information gathering.  Always obtain proper authorization before assessing any system. The contributors are not responsible for any illegal or unauthorized use.*

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:** Perform specific tasks (port scanning, vulnerability checks, credential brute-forcing) through independent modules.
*   **Multi-Protocol Support:** Scan a wide range of protocols including HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, and XML-RPC.
*   **Multithreaded Scanning:** Speed up assessments with parallel scanning capabilities.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database & Drift Detection:** Track scan history and detect changes in infrastructure, ideal for CI/CD.
*   **Flexible Interface:** Utilize CLI, REST API, and a web UI for control and analysis.
*   **Evasion Techniques:** Employ configurable delays, proxy support, and user-agent randomization to evade detection.
*   **Target Flexibility:** Supports single IPs, IP ranges, CIDR blocks, domain names, and URLs.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance, service discovery, and vulnerability scanning for efficient testing.
*   **Reconnaissance & Vulnerability Assessment:** Discover live hosts, open ports, services, and vulnerabilities.
*   **Attack Surface Mapping:** Quickly identify exposed assets, services, and subdomains.
*   **Bug Bounty Recon:** Automate subdomain enumeration, directory brute-forcing, and credential checks.
*   **Network Vulnerability Scanning:** Efficiently scan networks and detect vulnerabilities.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets using historical scan data.
*   **CI/CD & Compliance Monitoring:** Integrate Nettacker into pipelines to track infrastructure changes and vulnerabilities.

## Quick Setup & Run (Docker)

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

*   Access the Web GUI at https://localhost:5000.
*   Login using the API Key displayed in the CLI.
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` shares your nettacker folder, so you won't lose data after `docker-compose down`.
*   To view the API key, run `docker logs nettacker_nettacker`.
*   For more installation details without Docker: [Installation](https://nettacker.readthedocs.io/en/latest/Installation)

## Community and Resources

*   **Project Home Page:** [OWASP Nettacker](https://owasp.org/nettacker)
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

## Adopters

We're grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

See [ADOPTERS.md](ADOPTERS.md) for details.

## Google Summer of Code (GSoC) Project

*   ☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative
*   🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks:

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Contributing

OWASP Nettacker is an open-source project, built on the principles of collaboration and shared knowledge. We appreciate contributions from the community!

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)