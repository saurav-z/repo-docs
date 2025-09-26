# OWASP Nettacker: Automated Penetration Testing & Information Gathering Framework

**OWASP Nettacker is a powerful, open-source Python framework designed to automate penetration testing, reconnaissance, and vulnerability assessment, empowering security professionals to identify and address network and application weaknesses efficiently.** Explore the original repository [here](https://github.com/OWASP/Nettacker).

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

**DISCLAIMER:** *This software is for ethical use and penetration testing purposes only.  Ensure you have explicit permission before scanning any systems; the contributors are not responsible for misuse.*

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Key Features

*   **Modular Architecture:**  Execute specific tasks with independent modules, such as port scanning, directory discovery, and vulnerability checks, for granular control.
*   **Multi-Protocol & Multithreaded Scanning:**  Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scanning for improved speed.
*   **Comprehensive Reporting:** Export results in various formats, including HTML, JSON, CSV, and plain text.
*   **Built-in Database & Drift Detection:**  Store scan history for analysis and compare results to detect changes, useful for CI/CD pipelines.
*   **CLI, REST API & Web UI:** Offers flexibility through command-line interface, REST API, and a user-friendly web interface.
*   **Evasion Techniques:** Configure delays, proxy support, and randomized user-agents to avoid detection.
*   **Flexible Target Specification:** Accept single IPs, ranges, CIDR blocks, domain names, and URLs, with target lists.

## Use Cases

*   **Penetration Testing:** Automate reconnaissance and vulnerability assessments.
*   **Recon & Vulnerability Assessment:**  Discover live hosts, open ports, services, and default credentials.
*   **Attack Surface Mapping:**  Quickly identify exposed assets with built-in enumeration modules.
*   **Bug Bounty Recon:** Automate common reconnaissance tasks.
*   **Network Vulnerability Scanning:** Efficiently scan IP ranges and CIDR blocks.
*   **Shadow IT & Asset Discovery:** Uncover forgotten assets using scan history.
*   **CI/CD & Compliance Monitoring:**  Track infrastructure changes and detect new vulnerabilities.

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

*   Access the Web GUI at `https://localhost:5000` or `https://nettacker-api.z3r0d4y.com:5000/`
*   Use the API Key from the CLI to login.
*   The local database is `.nettacker/data/nettacker.db` (sqlite).
*   Default results path is `.nettacker/data/results`
*   `docker-compose` will share your nettacker folder, so you will not lose any data after `docker-compose down`
*   To see the API key in you can also run `docker logs nettacker_nettacker`.
*   More details and install without docker https://nettacker.readthedocs.io/en/latest/Installation

## Contributing

OWASP Nettacker is a collaborative project.  Contributions from the community are welcome!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## Adopters

We value the community's use of Nettacker. See [ADOPTERS.md](ADOPTERS.md) to learn more about how Nettacker is being utilized.

## Google Summer of Code

OWASP Nettacker has participated in the Google Summer of Code Initiative. Thanks to all the students who contributed!

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## Stargazers over time

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

<img alt="" referrerpolicy="no-referrer-when-downgrade" src="https://static.scarf.sh/a.png?x-pxid=8e922d16-445a-4c63-b4cf-5152fbbaf7fd" />