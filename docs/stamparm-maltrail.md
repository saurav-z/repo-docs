[![Maltrail](https://i.imgur.com/3xjInOD.png)](https://github.com/stamparm/maltrail)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Detect and Block Malicious Network Traffic

**Maltrail is an open-source, easy-to-use, and powerful malicious traffic detection system that protects your network by identifying and blocking threats in real-time.**

### Key Features

*   **Comprehensive Threat Detection:** Utilizes publicly available blacklists, static trails from AV reports, and user-defined lists to identify malicious traffic, including:
    *   Malware domains and URLs
    *   Suspicious IP addresses
    *   Malicious HTTP User-Agent strings
*   **Heuristic Analysis:** Includes advanced mechanisms to detect unknown and emerging threats.
*   **Real-Time Monitoring:**  Monitors network traffic passively or inline, providing immediate threat alerts.
*   **Flexible Architecture:**  Offers a modular design with a sensor, server, and client, allowing for various deployment options.
*   **Reporting Interface:**  Provides a user-friendly web interface for visualizing and analyzing detected threats.
*   **Customizable:** Supports user-defined trails and custom configurations to tailor detection to specific environments.
*   **Integration:**  Integrates with popular security tools such as Wazuh, Splunk, and XSOAR for enhanced threat response.

### Table of Contents

*   [Introduction](#introduction)
*   [Architecture](#architecture)
*   [Demo pages](#demo-pages)
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
    *   [Suspicious Ipinfo Requests](#suspicious-ipinfo-requests)
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

### Introduction

Maltrail is a robust malicious traffic detection system designed to identify and alert on potentially harmful network activity. It leverages both static and dynamic sources to identify threats. It uses public blacklists and AV reports, and can use static and custom trails.

[See Original README for Details](https://github.com/stamparm/maltrail)

### Architecture

Maltrail employs a flexible, layered architecture comprising:

*   **Traffic:** The network traffic that is being monitored.
*   **Sensor(s):**  Standalone components that passively monitor network traffic (SPAN/mirroring port, bridge, or standalone machine) for blacklisted items (domain names, URLs, IPs)
*   **Server:**  A central component that stores event details and provides a web-based reporting application.
*   **Client:**  The web interface that allows users to view and analyze detected threats.

![Architecture diagram](https://i.imgur.com/2IP9Mh2.png)

### Demo Pages

Explore real-life threat examples on the [demo pages](https://maltraildemo.github.io/).

### Requirements

*   Python 2.6, 2.7, or 3.x
*   pcapy-ng package installed
*   Sensor component requires at least 1GB of RAM, and root privileges.
*   Server component does not have any special requirements.

### Quick Start

Here's how to get Maltrail up and running on Ubuntu/Debian or SUSE/openSUSE:

*   **Ubuntu/Debian:**

    ```bash
    sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```

*   **SUSE/openSUSE:**

    ```bash
    sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```

*   Don't forget to put interfaces in promiscuous mode as needed: 

    ```bash
    for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
    ```

![Sensor](https://i.imgur.com/E9tt2ek.png)

To start the (optional) **Server** on same machine, open a new terminal and execute the following:

```bash
[[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
python server.py
```

![Server](https://i.imgur.com/loGW6GA.png)

*   **Docker:**

    *   Start the container(s) with `docker run`: 

    ```sh
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

Don't edit the `docker-compose.yml` file directly, as this will be overwritten by `git pull`.  Instead, copy it to `docker-compose.override.yml` and edit that file; it is included in this repo's `.gitignore`.  

To test that everything is up and running execute the following:

```sh
ping -c 1 136.161.101.53
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test](https://i.imgur.com/NYJg6Kl.png)

Also, to test the capturing of DNS traffic you can try the following:

```sh
nslookup morphed.ru
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test2](https://i.imgur.com/62oafEe.png)

To stop **Sensor** and **Server** instances (if running in background) execute the following:

```sh
sudo pkill -f sensor.py
pkill -f server.py
```

Access the reporting interface (i.e. **Client**) by visiting the http://127.0.0.1:8338 (default credentials: `admin:changeme!`) from your web browser:

![Reporting interface](https://i.imgur.com/VAsq8cs.png)

### Administrator's Guide

Refer to the [Administrator's Guide](#administrators-guide) for in-depth configuration options and deployment details.

### User's Guide

Refer to the [User's Guide](#users-guide) for detailed information on using the web interface.

### Real-Life Cases

*   [Mass Scans](#mass-scans)
*   [Anonymous Attackers](#anonymous-attackers)
*   [Service Attackers](#service-attackers)
*   [Malware](#malware)
*   [Suspicious Domain Lookups](#suspicious-domain-lookups)
*   [Suspicious Ipinfo Requests](#suspicious-ipinfo-requests)
*   [Suspicious Direct File Downloads](#suspicious-direct-file-downloads)
*   [Suspicious HTTP Requests](#suspicious-http-requests)
*   [Port Scanning](#port-scanning)
*   [DNS Resource Exhaustion](#dns-resource-exhaustion)
*   [Data Leakage](#data-leakage)
*   [False Positives](#false-positives)

### Best Practices

[See Best Practices Section for Details](#best-practices)

### License

[See License Section for Details](#license)

### Sponsors

[See Sponsors Section for Details](#sponsors)

### Developers

[See Developers Section for Details](#developers)

### Presentations

[See Presentations Section for Details](#presentations)

### Publications

[See Publications Section for Details](#publications)

### Blacklist

Maltrail uses a daily updated blacklist of malware-related domains.

[See Blacklist Section for Details](#blacklist)

### Thank You

[See Thank You Section for Details](#thank-you)

### Third-Party Integrations

Maltrail integrates with various security tools.

[See Third-Party Integrations Section for Details](#third-party-integrations)