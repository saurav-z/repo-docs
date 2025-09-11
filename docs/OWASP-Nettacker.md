# OWASP Nettacker: Your All-in-One Penetration Testing and Information Gathering Framework

**OWASP Nettacker** is a powerful, open-source framework designed to automate security assessments, helping you efficiently identify vulnerabilities in networks and web applications. ([Original Repository](https://github.com/OWASP/Nettacker))

[![Build Status](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)](https://github.com/OWASP/Nettacker/actions/workflows/ci_cd.yml/badge.svg?branch=master)
[![Apache License](https://img.shields.io/badge/License-Apache%20v2-green.svg)](https://github.com/OWASP/Nettacker/blob/master/LICENSE)
[![Twitter](https://img.shields.io/badge/Twitter-@iotscan-blue.svg)](https://twitter.com/iotscan)
![GitHub contributors](https://img.shields.io/github/contributors/OWASP/Nettacker)
[![Documentation Status](https://readthedocs.org/projects/nettacker/badge/?version=latest)](https://nettacker.readthedocs.io/en/latest/?badge=latest)
[![repo size ](https://img.shields.io/github/repo-size/OWASP/Nettacker)](https://github.com/OWASP/Nettacker)
[![Docker Pulls](https://img.shields.io/docker/pulls/owasp/nettacker)](https://hub.docker.com/r/owasp/nettacker)

<img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp-nettacker.png" width="200"><img src="https://raw.githubusercontent.com/OWASP/Nettacker/master/nettacker/web/static/img/owasp.png" width="500">

## **Key Features:**

*   **Modular Architecture:** Execute specific tasks such as port scanning, directory discovery, and vulnerability checks with individual modules for flexibility.
*   **Multi-Protocol and Multithreaded Scanning:** Supports HTTP/HTTPS, FTP, SSH, SMB, SMTP, ICMP, TELNET, XML-RPC, and parallel scans for speed.
*   **Comprehensive Reporting:** Generate reports in HTML, JSON, CSV, and plain text formats.
*   **Built-in Database and Drift Detection:** Store scan data for easy comparison and identification of changes over time.
*   **CLI, REST API & Web UI:**  Access the framework via command-line interface, REST API for integration, and a user-friendly web interface.
*   **Evasion Techniques:** Utilize delays, proxy support, and randomized user-agents to avoid detection.
*   **Flexible Target Input:** Scan single IPs, IP ranges, CIDR blocks, domain names, or URLs.

## **Use Cases:**

*   **Penetration Testing:** Automate reconnaissance and vulnerability scanning.
*   **Reconnaissance & Vulnerability Assessment:** Discover hosts, ports, services, and vulnerabilities.
*   **Attack Surface Mapping:** Quickly identify exposed assets.
*   **Bug Bounty Recon:** Automate reconnaissance tasks.
*   **Network Vulnerability Scanning:** Perform efficient assessments of networks.
*   **Shadow IT & Asset Discovery:** Uncover unmanaged assets through historical data.
*   **CI/CD & Compliance Monitoring:** Track infrastructure changes and detect new vulnerabilities.

## **Disclaimer**

***THIS SOFTWARE WAS CREATED FOR AUTOMATED PENETRATION TESTING AND INFORMATION GATHERING. YOU MUST USE THIS SOFTWARE IN A RESPONSIBLE AND ETHICAL MANNER. DO NOT TARGET SYSTEMS OR APPLICATIONS WITHOUT OBTAINING PERMISSIONS OR CONSENT FROM THE SYSTEM OWNERS OR ADMINISTRATORS. CONTRIBUTORS WILL NOT BE RESPONSIBLE FOR ANY ILLEGAL USAGE.***

![2018-01-19_0-45-07](https://user-images.githubusercontent.com/7676267/35123376-283d5a3e-fcb7-11e7-9b1c-92b78ed4fecc.gif)

## Quick Start with Docker

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

## **Contributions & Community**

OWASP Nettacker thrives on community contributions. Thank you to all the awesome contributors!

![Awesome Contributors](https://contrib.rocks/image?repo=OWASP/Nettacker)

## **Adopters**

We’re grateful to the organizations, community projects, and individuals who adopt and rely on OWASP Nettacker for their security workflows.

If you’re using OWASP Nettacker in your organization or project, we’d love to hear from you! Feel free to add your details to the [ADOPTERS.md](ADOPTERS.md) file by submitting a pull request or reach out to us via GitHub issues. Let’s showcase how Nettacker is making a difference in the security community!

 See [ADOPTERS.md](ADOPTERS.md) for details.

## **Google Summer of Code (GSoC) Project**

*	☀️ OWASP Nettacker Project is participating in the Google Summer of Code Initiative 
* 🙏 Thanks to Google Summer of Code Initiative and all the students who contributed to this project during their summer breaks: 

<a href="https://summerofcode.withgoogle.com"><img src="https://betanews.com/wp-content/uploads/2016/03/vertical-GSoC-logo.jpg" width="200"></img></a>

## **Stargazers over Time**

[![Stargazers over time](https://starchart.cc/OWASP/Nettacker.svg)](https://starchart.cc/OWASP/Nettacker)

## **Links**

*   OWASP Nettacker Project Home Page: https://owasp.org/nettacker
*   Documentation: https://nettacker.readthedocs.io
*   Slack: [#project-nettacker](https://owasp.slack.com/archives/CQZGG24FQ) on https://owasp.slack.com
*   Installation: https://nettacker.readthedocs.io/en/latest/Installation
*   Usage: https://nettacker.readthedocs.io/en/latest/Usage
*   GitHub repo: https://github.com/OWASP/Nettacker
*   Docker Image: https://hub.docker.com/r/owasp/nettacker
*   How to use the Dockerfile: https://nettacker.readthedocs.io/en/latest/Installation/#install-nettacker-using-docker
*   OpenHub: https://www.openhub.net/p/OWASP-Nettacker
*   **Donate**: https://owasp.org/donate/?reponame=www-project-nettacker&title=OWASP+Nettacker
*   **Read More**: https://www.secologist.com/open-source-projects