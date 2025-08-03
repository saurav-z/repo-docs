![Maltrail](https://i.imgur.com/3xjInOD.png)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Real-Time Malicious Traffic Detection

**Maltrail is a powerful and open-source network intrusion detection system that identifies malicious traffic using community-sourced blacklists and advanced heuristics.** [See the original repo](https://github.com/stamparm/maltrail)

**Key Features:**

*   **Real-Time Threat Detection:** Monitors network traffic for known threats and suspicious activities.
*   **Comprehensive Blacklist Integration:** Utilizes a vast array of public and static blacklists.
*   **Heuristic Analysis:** Employs advanced techniques to identify unknown threats.
*   **Flexible Architecture:** Operates with a sensor, server, and client setup, or as a standalone sensor.
*   **User-Friendly Reporting Interface:** Provides an intuitive web interface for threat analysis and reporting.
*   **Customizable Rules:** Allows users to add custom trails for specific threat monitoring.
*   **Third-Party Integrations:** Offers integrations with popular security tools and platforms.

**Key Functionality Breakdown:**

*   **Introduction:** Maltrail is a malicious traffic detection system.
*   **Architecture:** Utilizes a "Traffic -> Sensor <-> Server <-> Client" architecture.  Sensors monitor network traffic for malicious indicators, forwarding alerts to the server, where reporting can be performed on the client side.
*   **Demo Pages:** Provides functional demo pages to showcase real-life threats.
*   **Requirements:** Requires Python 2.6, 2.7, or 3.x with pcapy-ng.
*   **Quick Start:** Simple commands to get the sensor and server up and running.
*   **Administrator's Guide:** Explains how to configure the sensor and server.
*   **User's Guide:** Covers the reporting interface and its features.
*   **Real-Life Cases:** Demonstrates Maltrail's capabilities in identifying various threats.

**Detailed Sections:**

*   [Administrator's guide](#administrators-guide)
    *   [Sensor](#sensor)
    *   [Server](#server)
*   [User's guide](#users-guide)
    *   [Reporting interface](#reporting-interface)
*   [Real-life cases](#real-life-cases)
    *   [Mass scans](#mass-scans)
    *   [Anonymous attackers](#anonymous-attackers)
    *   [Service attackers](#service-attackers)
    *   [Malware](#malware)
    *   [Suspicious domain lookups](#suspicious-domain-lookups)
    *   [Suspicious ipinfo requests](#suspicious-ipinfo-requests)
    *   [Suspicious direct file downloads](#suspicious-direct-file-downloads)
    *   [Suspicious HTTP requests](#suspicious-http-requests)
    *   [Port scanning](#port-scanning)
    *   [DNS resource exhaustion](#dns-resource-exhaustion)
    *   [Data leakage](#data-leakage)
    *   [False positives](#false-positives)
*   [Best practice(s)](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank you](#thank-you)
*   [Third-party integrations](#third-party-integrations)

**Get Started Today:**

Follow the quick start guide above to quickly begin detecting malicious network traffic!