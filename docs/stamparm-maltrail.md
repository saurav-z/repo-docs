# Maltrail: Detect and Mitigate Malicious Network Traffic

Maltrail is a powerful and open-source network traffic detection system that proactively identifies and blocks malicious activity, helping you safeguard your network. Learn more and contribute at [https://github.com/stamparm/maltrail](https://github.com/stamparm/maltrail).

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Key Features

*   **Real-time Threat Detection:** Identifies malicious traffic using blacklists, static trails, and heuristic analysis.
*   **Comprehensive Data Feeds:** Leverages a wide array of publicly available threat intelligence feeds and static trails.
*   **Flexible Architecture:** Works with a **Sensor**, **Server**, and **Client** architecture, offering deployment flexibility.
*   **User-Friendly Interface:** Provides a clear and concise reporting interface for easy threat analysis.
*   **Customizable:** Allows users to define custom trails and integrate with third-party tools.
*   **Extensive Malware Coverage:** Includes detection for a wide range of malware families and attack techniques.
*   **Open Source & Community-Driven:** Benefit from continuous improvements and updates from a dedicated community.

## Table of Contents

*   [Introduction](#introduction)
*   [Architecture](#architecture)
*   [Demo Pages](#demo-pages)
*   [Quick Start](#quick-start)
*   [Administrator's Guide](#administrators-guide)
    *   [Sensor](#sensor)
    *   [Server](#server)
*   [User's Guide](#users-guide)
    *   [Reporting Interface](#reporting-interface)
*   [Real-Life Cases](#real-life-cases)
*   [Best Practices](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank You](#thank-you)
*   [Third-Party Integrations](#third-party-integrations)

## Introduction

**Maltrail** is a malicious traffic detection system that identifies and alerts you to suspicious activities. It works by using publicly available blacklists containing malicious and/or suspicious entries. These entries can be domain names (e.g., `zvpprsensinaix.com` for Banjori malware), URLs (e.g., `hXXp://109.162.38.120/harsh02.exe` for a known malicious executable), IP addresses (e.g., `185.130.5.231` for a known attacker), or HTTP User-Agent headers (e.g., `sqlmap` for SQL injection tools).  Maltrail also uses advanced heuristics to help detect new and unknown threats.

Maltrail utilizes numerous blacklists (feeds) and also includes static trails for known malicious entities to provide comprehensive threat detection.

![Reporting tool](https://i.imgur.com/Sd9eqoa.png)

## Architecture

Maltrail utilizes a **Traffic** -> **Sensor** <-> **Server** <-> **Client** architecture. The **Sensor** component monitors network traffic for malicious indicators. It can run on the same machine as the Server, or on a separate device connected to a SPAN/mirroring port, or in a transparent inline bridge. When a threat is detected, the Sensor sends event details to the central **Server**, which stores the data. The **Server** then provides a back-end for the reporting web application. Data is transferred to the client and processed in the web browser, allowing for quick and efficient presentation of events.

![Architecture diagram](https://i.imgur.com/2IP9Mh2.png)

**Note:**  The **Server** component can be omitted. In this case, the **Sensor** stores events locally, which can be examined manually or by other tools.

## Demo Pages

Explore the functionality of Maltrail with fully functional demo pages showcasing real-life threats: [https://maltraildemo.github.io/](https://maltraildemo.github.io/).

## Quick Start

Quickly get Maltrail up and running with the following commands.

**Requirements:**
*   Python (2.6, 2.7 or 3.x)
*   pcapy-ng package

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

**Don't forget to put interfaces in promiscuous mode as needed:**

```bash
for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
```

![Sensor](https://i.imgur.com/E9tt2ek.png)

**To start the (optional) Server on the same machine:**

```bash
[[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
python server.py
```

![Server](https://i.imgur.com/loGW6GA.png)

**Docker:**

```bash
# Build image
# Start the server
docker run -d --name maltrail --restart=unless-stopped --port 8338:8338/tcp --port 8337:8337/udp -v /etc/maltrail.conf:/opt/maltrail/maltrail.conf:ro ghcr.io/stamparm/maltrail:latest
# Update the image regularly
docker stop maltrail
docker pull ghcr.io/stamparm/maltrail:latest
docker start maltrail
```

**Test:**

```bash
ping -c 1 136.161.101.53
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test](https://i.imgur.com/NYJg6Kl.png)

**DNS traffic test:**

```bash
nslookup morphed.ru
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test2](https://i.imgur.com/62oafEe.png)

**Stop Sensor and Server:**

```bash
sudo pkill -f sensor.py
pkill -f server.py
```

**Reporting interface access:**

*   Open a web browser and visit http://127.0.0.1:8338 (default credentials: `admin:changeme!`).

![Reporting interface](https://i.imgur.com/VAsq8cs.png)

## Administrator's Guide

### Sensor

The Sensor's configuration settings are in the `maltrail.conf` file's `[Sensor]` section.

![Sensor's configuration](https://i.imgur.com/8yZKH14.png)

*   `USE_MULTIPROCESSING`: Enables multi-core processing.
*   `USE_FEED_UPDATES`: Enables/Disables trail updates from feeds.
*   `UPDATE_PERIOD`: Sets the interval (in seconds) for automatic trail updates.
*   `CUSTOM_TRAILS_DIR`: Allows you to specify a directory containing custom trail files.
*   `USE_HEURISTICS`: Activates heuristic mechanisms (may introduce false positives).
*   `CAPTURE_BUFFER`:  Specifies the memory to use for storing packet capture in multiprocessing mode.
*   `MONITOR_INTERFACE`: Defines the interface to capture traffic from.  Use "any" to monitor all interfaces.
*   `CAPTURE_FILTER`: Allows you to specify a tcpdump filter.
*   `SENSOR_NAME`: Sets the sensor name for event identification.
*   `LOG_SERVER`: If set, sends events remotely to the Server.  If not, events are stored locally.
*   `UPDATE_SERVER`: Specifies a server to pull trail updates from.
*   `SYSLOG_SERVER` / `LOGSTASH_SERVER`: Allows events to be sent to non-Maltrail servers in CEF or JSON format.

### Server

Server configuration settings are found in the `maltrail.conf`'s `[Server]` section.

![Server's configuration](https://i.imgur.com/TiUpLX8.png)

*   `HTTP_ADDRESS`: Specifies the web server's listening address (use 0.0.0.0 for all interfaces).
*   `HTTP_PORT`: Sets the web server's listening port.  Defaults to 8338.
*   `USE_SSL`:  Enables SSL/TLS for HTTPS access to the web server.
*   `SSL_PEM`: Specifies the path to the server's private/cert PEM file when using SSL.
*   `USERS`: Contains user configuration settings. Each user is defined as `username:sha256(password):UID:filter_netmask(s)`.
*   `UDP_ADDRESS`: Specifies the server's log collecting listening address (use 0.0.0.0 for all interfaces).
*   `UDP_PORT`: Sets the UDP listening port.
*   `FAIL2BAN_REGEX`: Allows regular expressions to be used in `/fail2ban` web calls for IP blocking.
*   `BLACKLIST`: Allows regular expressions to be created to apply to specific fields.
*   `BLACKLIST_NAME`: (Optional) Allows name of the blacklist to be added, which will build the URL : `/blacklist/<name>`.

## User's Guide

### Reporting Interface

Access the reporting interface using the address defined in the `HTTP_ADDRESS` and `HTTP_PORT` options within the configuration file.  Enter your credentials to login (default: `admin:changeme!`).

![User login](https://i.imgur.com/WVpASAI.png)

Once logged in, you'll be presented with the reporting interface:

![Reporting interface](https://i.imgur.com/PZY8JEC.png)

The interface provides a sliding timeline for selecting past events and a summary of displayed events. The bottom of the interface provides a table of all threats:

![Single threat](https://i.imgur.com/IxPwKKZ.png)

*   `threat`:  Threat's unique ID and color
*   `sensor`: Sensor name(s)
*   `events`:  Total number of events for a threat
*   `severity`: Evaluated threat severity.
*   `first_seen`: Time of the first event.
*   `last_seen`: Time of the last event.
*   `sparkline`: Activity graph for the threat.
*   `src_ip`: Source IPs
*   `src_port`: Source ports
*   `dst_ip`: Destination IPs
*   `dst_port`: Destination ports
*   `proto`: Protocols
*   `trail`:  Blacklisted entry that triggered the event(s).
*   `info`: More information about the threat/trail.
*   `reference`: Source of the blacklisted entry.
*   `tags`: User-defined tags.

Tooltips for IPs and condensed port details are available by hovering over the respective icons. Hovering over a `trail` entry displays search results in a frame.
Tagging threats allows for categorization and filtering.

## Real-Life Cases

Maltrail provides effective detection of real-world threats.
Detailed examples are in the original README.md

## Best Practice(s)

Follow the steps for optimal Maltrail installation, environment setup, autostart, and systemd service configuration.
Detailed examples are in the original README.md

## License

This software is available under the MIT License.  See the [LICENSE](https://github.com/stamparm/maltrail/blob/master/LICENSE) file for details.

## Sponsors

*   [Sansec](https://sansec.io/) (2024-)
*   [Sansec](https://sansec.io/) (2020-2021)

## Developers

*   Miroslav Stampar ([@stamparm](https://github.com/stamparm))
*   Mikhail Kasimov ([@MikhailKasimov](https://github.com/MikhailKasimov))

## Presentations

*   47th TF-CSIRT Meeting, Prague (Czech Republic), 2016 ([slides](https://www.terena.org/activities/tf-csirt/meeting47/M.Stampar-Maltrail.pdf))

## Publications

*   Detect attacks on your network with Maltrail, Linux Magazine, 2022 ([Annotation](https://www.linux-magazine.com/Issues/2022/258/Maltrail))
*   Best Cyber Threat Intelligence Feeds ([SilentPush Review, 2022](https://www.silentpush.com/blog/best-cyber-threat-intelligence-feeds))
*   Research on Network Malicious Traffic Detection System Based on Maltrail ([Nanotechnology Perceptions, ISSN 1660-6795, 2024](https://nano-ntp.com/index.php/nano/article/view/1915/1497))

## Blacklist

Find Maltrail's updated malware-related domain blacklist at [https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt](https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt).

## Thank You

*   Thomas Kristner
*   Eduardo Arcusa Les
*   James Lay
*   Ladislav Baco (@laciKE)
*   John Kristoff (@jtkdpu)
*   Michael M&uuml;nz (@mimugmail)
*   David Brush
*   @Godwottery
*   Chris Wild (@briskets)
*   Keith Irwin (@ki9us)
*   Simon Szustkowski (@simonszu)

## Third-Party Integrations

A list of third-party integrations is included in the original README.md.