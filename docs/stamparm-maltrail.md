[![Maltrail](https://i.imgur.com/3xjInOD.png)](https://github.com/stamparm/maltrail)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Detect Malicious Traffic and Protect Your Network

**Maltrail is a powerful, open-source malicious traffic detection system that uses blacklists and heuristics to identify and alert on suspicious network activity.**

### Key Features

*   **Real-time Threat Detection:** Monitors network traffic for malicious indicators.
*   **Blacklist Integration:** Leverages multiple public and user-defined blacklists, including domains, URLs, IPs, and User-Agent headers.
*   **Heuristic Analysis:** Employs advanced heuristics to detect unknown threats and suspicious patterns.
*   **Flexible Architecture:** Supports a modular architecture with sensors, a server, and a reporting interface.
*   **Comprehensive Reporting:** Provides a web-based interface for visualizing and analyzing detected threats.
*   **Third-Party Integrations:** Integrates with various security tools and platforms.
*   **Customizable:** Allows for custom trails and configurations to match your specific needs.

### Table of Contents

*   [Introduction](#introduction)
*   [Key Features](#key-features)
*   [Architecture](#architecture)
*   [Demo Pages](#demo-pages)
*   [Requirements](#requirements)
*   [Quick Start](#quick-start)
*   [Administrator's Guide](#administrators-guide)
    *   [Sensor](#sensor)
    *   [Server](#server)
*   [User's Guide](#users-guide)
    *   [Reporting Interface](#reporting-interface)
*   [Real-Life Cases](#real-life-cases)
    *   [Mass Scans](#mass-scans)
    *   [Anonymous Attackers](#anonymous-attackers)
    *   [Service Attackers](#service-attackers)
    *   [Malware](#malware)
    *   [Suspicious Domain Lookups](#suspicious-domain-lookups)
    *   [Suspicious IPinfo Requests](#suspicious-ipinfo-requests)
    *   [Suspicious Direct File Downloads](#suspicious-direct-file-downloads)
    *   [Suspicious HTTP Requests](#suspicious-http-requests)
    *   [Port Scanning](#port-scanning)
    *   [DNS Resource Exhaustion](#dns-resource-exhaustion)
    *   [Data Leakage](#data-leakage)
    *   [False Positives](#false-positives)
*   [Best Practices](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank You](#thank-you)
*   [Third-Party Integrations](#third-party-integrations)

### Architecture

Maltrail's architecture follows a **Traffic** -> **Sensor** <-> **Server** <-> **Client** model.

*   **Sensor(s):** Standalone component that monitors network traffic for malicious trails (domains, URLs, IPs, etc.). Upon detection, events are sent to the Server.
*   **Server:** Stores event details and provides back-end support for the reporting web application.
*   **Client:** Reporting web application responsible for data presentation and analysis.

![Architecture Diagram](https://i.imgur.com/2IP9Mh2.png)

### Demo Pages

Explore real-life threat examples on the demo pages: [here](https://maltraildemo.github.io/).

### Requirements

*   Python 2.6, 2.7 or 3.x
*   pcapy-ng package
    
    **NOTE:** Use of ```pcapy``` lib instead of ```pcapy-ng``` can lead to incorrect work of Maltrail, especially on **Python 3.x** environments. [Examples](https://github.com/stamparm/maltrail/issues?q=label%3Apcapy-ng-related+is%3Aclosed).

*   Sensor component requires at least 1GB of RAM to run in single-process mode or more if run in multiprocessing mode, depending on the value used for option `CAPTURE_BUFFER`.
*   Sensor component requires administrative/root privileges.
*   Server component does not have any special requirements.

### Quick Start

Follow these steps to get Maltrail up and running:

**Ubuntu/Debian:**

```bash
sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

**SUSE/openSUSE:**

```bash
sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

Put interfaces in promiscuous mode:

```bash
for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
```

**Start the Server:** Open a new terminal:

```bash
[[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
python server.py
```

**Docker:**

```bash
# Build image
docker build -t maltrail .
# Start the server
docker run -d --name maltrail-server --restart=unless-stopped --port 8338:8338/tcp --port 8337:8337/udp -v /etc/maltrail.conf:/opt/maltrail/maltrail.conf:ro maltrail
# Update the image regularly
sudo git pull
docker build -t maltrail .
```

... or with `docker compose`:

```sh
# For the sensor
docker compose up -d sensor
# For the server
docker compose up -d server
# For both
docker compose up -d
# Update image regularly
docker compose down --remove-orphans
docker compose build
docker compose up -d
```

To test:

```bash
ping -c 1 136.161.101.53
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

DNS test:

```bash
nslookup morphed.ru
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

**Stop Sensor and Server:**

```bash
sudo pkill -f sensor.py
pkill -f server.py
```

**Access the Reporting Interface:**  Open your web browser and go to: `http://127.0.0.1:8338` (default credentials: `admin:changeme!`).

[See Original Repo for More Details](https://github.com/stamparm/maltrail)