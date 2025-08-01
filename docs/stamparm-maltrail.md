![Maltrail](https://i.imgur.com/3xjInOD.png)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Detect Malicious Traffic with Open-Source Intelligence

**Maltrail is a powerful and open-source malicious traffic detection system that utilizes public and custom blacklists to identify and alert on threats in your network.**  ([Original Repo](https://github.com/stamparm/maltrail))

### Key Features

*   **Real-time Monitoring:** Continuously monitors network traffic for malicious indicators.
*   **Comprehensive Threat Detection:** Uses a wide range of blacklists, static trails, and heuristic analysis to identify threats.
*   **Flexible Architecture:**  Supports a Traffic -> Sensor -> Server -> Client architecture for scalable deployment.  Also works with standalone sensor mode.
*   **Customizable Rules:** Allows users to define custom trails and integrate with third-party tools.
*   **User-Friendly Interface:** Provides a web-based reporting interface for easy threat analysis and incident response.
*   **Docker Support**: Easily deploy Maltrail using Docker containers.

### Getting Started

*   **Quick Installation:** Detailed instructions are provided for Ubuntu/Debian, SUSE/openSUSE, and Docker environments.

*   **Core Components:**
    *   **Sensor:**  The core of Maltrail, it runs on the monitoring node and analyzes network traffic based on configured trails and heuristics.
    *   **Server:**  (Optional) Stores event details and provides a web-based reporting interface.
    *   **Client:**  A web-based reporting interface for viewing and analyzing threat data.

*   **Basic Commands**:

    **For Ubuntu/Debian**

    ```sh
    sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```

    **For SUSE/openSUSE**

    ```sh
    sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```

    **For Docker**
    ```sh
    # Build image
    docker build -t maltrail .
    # Start the server
    docker run -d --name maltrail-server --restart=unless-stopped --port 8338:8338/tcp --port 8337:8337/udp -v /etc/maltrail.conf:/opt/maltrail/maltrail.conf:ro maltrail
    # Update the image regularly
    sudo git pull
    docker build -t maltrail .
    ```
### Administrator's Guide

*   **Sensor Configuration:** The `maltrail.conf` file configures the sensor's behavior, including interfaces to monitor, logging, and heuristic settings.
*   **Server Configuration:** The `maltrail.conf` file configures the server including  HTTP address and port, SSL/TLS, user accounts and fail2ban integration.
*   **Configuration Files:** Location and descriptions for the configuration files.
*   **Log Analysis:**  Logs are stored in CSV format,  facilitating easy analysis.

### User's Guide

*   **Reporting Interface:**  Access the interface through a web browser to view real-time alerts.
*   **Interactive Dashboard:** Features a timeline for historical data, interactive charts for threat analysis, and a detailed table of events.
*   **Advanced Features:** IP and DNS information tooltips and search integration
### Real-Life Cases

*   **Mass Scans:** Detects common scanning activities.
*   **Anonymous Attackers:** Identifies threats using Tor exit nodes.
*   **Service Attacks:** Monitors for attacks against specific services.
*   **Malware Detection:**  Identifies connections to known C&C servers.
*   **Suspicious Domain Lookups:**  Detects DGA domains and other suspicious domain activity.
*   **Data Leakage:**  Highlights potential data exfiltration attempts.

### Best Practices

*   Provides a recommended set of steps.
*   Enabling systemd services (Linux only).

### Resources

*   [License](https://github.com/stamparm/maltrail/blob/master/LICENSE)
*   [Sponsors](https://github.com/stamparm/maltrail#sponsors)
*   [Developers](https://github.com/stamparm/maltrail#developers)
*   [Presentations](https://github.com/stamparm/maltrail#presentations)
*   [Publications](https://github.com/stamparm/maltrail#publications)
*   [Blacklist](https://github.com/stamparm/aux/blob/master/maltrail-malware-domains.txt)
*   [Thank You](https://github.com/stamparm/maltrail#thank-you)
*   [Third-party integrations](https://github.com/stamparm/maltrail#third-party-integrations)