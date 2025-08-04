![Maltrail](https://i.imgur.com/3xjInOD.png)

[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Your First Line of Defense Against Malicious Network Traffic

Maltrail is a powerful, open-source network traffic detection system designed to identify and block malicious activity by utilizing blacklists, static trails, and heuristic analysis.  Visit the [original repository](https://github.com/stamparm/maltrail) for more details.

**Key Features:**

*   **Real-time Threat Detection:** Monitors network traffic for malicious domains, URLs, IPs, and User-Agent strings.
*   **Multiple Data Sources:** Leverages a vast array of public blacklists and static trails from AV reports and custom lists.
*   **Heuristic Analysis:** Employs advanced techniques to identify unknown threats.
*   **Flexible Architecture:**  Works with standalone sensors or a centralized server/client setup.
*   **User-Friendly Reporting Interface:** Provides a web-based dashboard for visualizing and analyzing threats.
*   **Easy Deployment:** Simple setup with clear installation instructions for various platforms.
*   **Extensive List Support**: Tracks 1494 Malware Families and 1354 Malware Sinkholes.
*   **Open Source:**  Free to use and modify under the MIT License.

**Table of Contents:**

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

## Introduction

Maltrail is a malicious traffic detection system, designed to identify and block malicious activity by utilizing publicly available (black)lists, static trails, and heuristic analysis.  Trails can include domain names (e.g. `zvpprsensinaix.com` for [Banjori](http://www.johannesbader.ch/2015/02/the-dga-of-banjori/) malware), URLs (e.g. `hXXp://109.162.38.120/harsh02.exe` for known malicious [executable](https://www.virustotal.com/en/file/61f56f71b0b04b36d3ef0c14bbbc0df431290d93592d5dd6e3fffcc583ec1e12/analysis/)), IP addresses (e.g. `185.130.5.231` for known attacker), or HTTP User-Agent header values (e.g. `sqlmap` for automatic SQL injection and database takeover tool). Additionally, it uses optional advanced heuristic mechanisms to discover unknown threats (e.g. new malware).

![Reporting tool](https://i.imgur.com/Sd9eqoa.png)

The following (black)lists (i.e. feeds) are being utilized:

```
360bigviktor, 360chinad, 360conficker, 360cryptolocker, 360gameover, 
360locky, 360necurs, 360suppobox, 360tofsee, 360virut, abuseipdb, alienvault, 
atmos, badips, bitcoinnodes, blackbook, blocklist, botscout, 
bruteforceblocker, ciarmy, cobaltstrike, cruzit, cybercrimetracker, 
dataplane, dshieldip, emergingthreatsbot, emergingthreatscip, 
emergingthreatsdns, feodotrackerip, gpfcomics, greensnow, ipnoise,
kriskinteldns, kriskintelip, malc0de, malwaredomainlistdns, malwaredomains,
maxmind, minerchk, myip, openphish, palevotracker, policeman, pony,
proxylists, proxyrss, proxyspy, ransomwaretrackerdns, ransomwaretrackerip, 
ransomwaretrackerurl, riproxies, rutgers, sblam, socksproxy, sslbl, 
sslproxies, talosintelligence, torproject, trickbot, turris, urlhaus, 
viriback, vxvault, zeustrackermonitor, zeustrackerurl, etc.
```

As for static entries, the trails for the following malicious entities (e.g. malware C&Cs or sinkholes) have been manually included (from various AV reports and personal research):

```
1ms0rry, 404, 9002, aboc, absent, ab, acbackdoor, acridrain, activeagent, 
adrozek, advisorbot, adwind, adylkuzz, adzok, afrodita, agaadex, agenttesla, 
aldibot, alina, allakore, almalocker, almashreq, alpha, alureon, amadey, 
amavaldo, amend_miner, ammyyrat, android_acecard, android_actionspy, 
android_adrd, android_ahmythrat, android_alienspy, android_andichap, 
android_androrat, android_anubis, android_arspam, android_asacub, 
android_backflash, android_bankbot, android_bankun, android_basbanke, 
android_basebridge, android_besyria, android_blackrock, android_boxer, 
android_buhsam, android_busygasper, android_calibar, android_callerspy, 
android_camscanner, android_cerberus, android_chuli, android_circle, 
android_claco, android_clickfraud, android_cometbot, android_cookiethief, 
android_coolreaper, android_copycat, android_counterclank, android_cyberwurx, 
android_darkshades, android_dendoroid, android_dougalek, android_droidjack, 
android_droidkungfu, android_enesoluty, android_eventbot, android_ewalls, 
android_ewind, android_exodus, android_exprespam, android_fakeapp, 
android_fakebanco, android_fakedown, android_fakeinst, android_fakelog, 
android_fakemart, android_fakemrat, android_fakeneflic, android_fakesecsuit, 
android_fanta, android_feabme, android_flexispy, android_fobus, 
android_fraudbot, android_friend, android_frogonal, android_funkybot, 
android_gabas, android_geinimi, android_generic, android_geost, 
android_ghostpush, android_ginmaster, android_ginp, android_gmaster, 
android_gnews, android_godwon, android_golddream, android_goldencup, 
android_golfspy, android_gonesixty, android_goontact, android_gplayed, 
android_gustuff, android_gypte, android_henbox, android_hiddad, 
android_hydra, android_ibanking, android_joker, android_jsmshider, 
android_kbuster, android_kemoge, android_ligarat, android_lockdroid, 
android_lotoor, android_lovetrap, android_malbus, android_mandrake, 
android_maxit, android_mobok, android_mobstspy, android_monokle, 
android_notcompatible, android_oneclickfraud, android_opfake, 
android_ozotshielder, android_parcel, android_phonespy, android_pikspam, 
android_pjapps, android_qdplugin, android_raddex, android_ransomware, 
android_redalert, android_regon, android_remotecode, android_repane, 
android_riltok, android_roamingmantis, android_roidsec, android_rotexy, 
android_samsapo, android_sandrorat, android_selfmite, android_shadowvoice, 
android_shopper, android_simbad, android_simplocker, android_skullkey, 
android_sndapps, android_spynote, android_spytekcell, android_stels, 
android_svpeng, android_swanalitics, android_teelog, android_telerat, 
android_tetus, android_thiefbot, android_tonclank, android_torec, 
android_triada, android_uracto, android_usbcleaver, android_viceleaker, 
android_vmvol, android_walkinwat, android_windseeker, android_wirex, 
android_wolfrat, android_xavirad, android_xbot007, android_xerxes, 
android_xhelper, android_xploitspy, android_z3core, android_zertsecurity, 
android_ztorg, andromeda, antefrigus, antibot, anubis, anuna, apocalypse, 
apt_12, apt_17, apt_18, apt_23, apt_27, apt_30, apt_33, apt_37, apt_38, 
apt_aridviper, apt_babar, apt_bahamut, etc.
```

## Architecture

Maltrail utilizes a **Traffic** -> **Sensor** <-> **Server** <-> **Client** architecture. The **Sensor**(s) is a standalone component that runs on a monitoring node (e.g., a Linux system connected passively to a SPAN/mirroring port or transparently inline on a Linux bridge) or a standalone machine (e.g., a Honeypot). It monitors incoming **Traffic** for blacklisted items/trails (i.e., domain names, URLs, and/or IPs). When a match is found, it sends event details to the (central) **Server**, where they are stored in the appropriate logging directory (i.e., `LOG_DIR` described in the *Configuration* section). If the **Sensor** and **Server** run on the same machine (default configuration), logs are stored directly in the local logging directory. Otherwise, they are sent via UDP messages to the remote server (i.e., `LOG_SERVER` described in the *Configuration* section).

![Architecture diagram](https://i.imgur.com/2IP9Mh2.png)

The **Server** primarily stores event details and provides back-end support for the reporting web application.  In the default configuration, the server and sensor will run on the same machine. The front-end reporting part is based on the ["Fat client"](https://en.wikipedia.org/wiki/Fat_client) architecture (i.e., all data post-processing is done inside the client's web browser instance). Events (i.e., log entries) for the selected (24h) period are transferred to the **Client**, where the reporting web application handles the presentation. Data is sent to the client in compressed chunks, and processed sequentially. The final report is created in a highly condensed form, allowing the presentation of a virtually unlimited number of events.

**Note:**  The **Server** component can be skipped entirely, using only the standalone **Sensor**. In this case, all events are stored in the local logging directory, and log entries can be examined manually or with a CSV reading application.

## Demo pages

Fully functional demo pages with collected real-life threats can be found [here](https://maltraildemo.github.io/).

## Requirements

To run Maltrail, [Python](http://www.python.org/download/) **2.6**, **2.7** or **3.x** is required on a \*nix/BSD system, together with the installed [pcapy-ng](https://pypi.org/project/pcapy-ng/) package.

**NOTE:** Use of ```pcapy``` lib instead of ```pcapy-ng``` can lead to incorrect work of Maltrail, especially on **Python 3.x** environments. [Examples](https://github.com/stamparm/maltrail/issues?q=label%3Apcapy-ng-related+is%3Aclosed).

*   **Sensor**:  Requires at least 1GB of RAM in single-process mode, or more if running in multiprocessing mode, depending on the `CAPTURE_BUFFER` setting. The **Sensor** component (in the general case) also requires administrative/root privileges.
*   **Server**:  No special requirements.

## Quick Start

The following commands will get your Maltrail **Sensor** up and running (using default settings and monitoring the "any" interface):

*   **Ubuntu/Debian**:

```bash
sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

*   **SUSE/openSUSE**:

```bash
sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
sudo pip3 install pcapy-ng
git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
sudo python3 sensor.py
```

Don't forget to put interfaces in promiscuous mode as needed:

```bash
for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
```

![Sensor](https://i.imgur.com/E9tt2ek.png)

To start the (optional) **Server** on the same machine, open a new terminal and execute:

```bash
[[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
cd maltrail
python server.py
```

![Server](https://i.imgur.com/loGW6GA.png)

*   **Docker**:

Download maltrail:

```sh
cd /usr/local/src
sudo git clone https://github.com/stamparm/maltrail.git
cd maltrail
sudo wget -P /etc https://raw.githubusercontent.com/stamparm/maltrail/master/maltrail.conf
# Edit the config
sudo $EDITOR /etc/maltrail.conf
```

Start the container(s) with `docker run`:

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

To test if everything is running, execute:

```bash
ping -c 1 136.161.101.53
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test](https://i.imgur.com/NYJg6Kl.png)

Also, to test the capturing of DNS traffic, you can try:

```bash
nslookup morphed.ru
cat /var/log/maltrail/$(date +"%Y-%m-%d").log
```

![Test2](https://i.imgur.com/62oafEe.png)

To stop **Sensor** and **Server** instances (if running in the background), execute:

```bash
sudo pkill -f sensor.py
pkill -f server.py
```

Access the reporting interface (i.e., **Client**) by visiting http://127.0.0.1:8338 (default credentials: `admin:changeme!`) in your web browser:

![Reporting interface](https://i.imgur.com/VAsq8cs.png)

## Administrator's Guide

### Sensor

Sensor configuration can be found inside the `maltrail.conf` file's `[Sensor]` section:

![Sensor's configuration](https://i.imgur.com/8yZKH14.png)

*   If `USE_MULTIPROCESSING` is `true`, all CPU cores will be used. One core is used for packet capture (with appropriate affinity, IO priority, and nice level settings), while others are used for packet processing.
*   `USE_FEED_UPDATES` can disable trail updates from feeds.
*   `UPDATE_PERIOD` (default: `86400` seconds or one day) controls automatic trail updates using definitions in the `trails` directory.
*   `CUSTOM_TRAILS_DIR` specifies the location of custom trail (`*.txt`) files.
*   `USE_HEURISTICS` enables heuristic mechanisms, potentially increasing false positives.
*   `CAPTURE_BUFFER` defines memory (in bytes or percentage) for storing packet capture in a ring buffer when using multiprocessing.
*   `MONITOR_INTERFACE` specifies the capturing interface (use `any` to capture from all interfaces if supported).
*   `CAPTURE_FILTER` contains the network capture (`tcpdump`) filter to exclude uninteresting packets.
*   `SENSOR_NAME` sets the name that appears in the events' `sensor_name` field, to distinguish events from different sensors.
*   `LOG_SERVER`:  If set, events are sent remotely to the **Server**; otherwise, they're stored directly in the logging directory specified with `LOG_DIR` (in the `[All]` section of `maltrail.conf`).
*   `UPDATE_SERVER` specifies the location to pull trails from; otherwise, they are updated from definitions in the installation.
*   `SYSLOG_SERVER` and/or `LOGSTASH_SERVER` can be used to send sensor events to non-Maltrail servers. Events will be sent to Syslog or Logstash in CEF (Common Event Format) or JSON format, respectively.

    *   For `SYSLOG_SERVER` (note: `LogSeverity` values are 0 (low), 1 (medium), and 2 (high)):

```
Dec 24 15:05:55 beast CEF:0|Maltrail|sensor|0.27.68|2020-12-24|andromeda (malware)|2|src=192.168.5.137 spt=60453 dst=8.8.8.8 dpt=53 trail=morphed.ru ref=(static)
```

    *   For `LOGSTASH_SERVER`:

```json
{"timestamp": 1608818692, "sensor": "beast", "severity": "high", "src_ip": "192.168.5.137", "src_port": 48949, "dst_ip": "8.8.8.8", "dst_port": 53, "proto": "UDP", "type": "DNS", "trail": "morphed.ru", "info": "andromeda (malware)", "reference": "(static)"}
```

When running the sensor for the first time or after a long period, it automatically updates trails. After initialization, it monitors the configured interface and writes events to either the logging directory or sends them remotely to the logging/reporting **Server**.

![Sensor run](https://i.imgur.com/A0qROp8.png)

Detected events are stored inside the **Server**'s logging directory (i.e. `LOG_DIR` inside the `maltrail.conf` file's section `[All]`) in easy-to-read CSV format (Note: whitespace ' ' is used as a delimiter) as single line entries consisting of: `time` `sensor` `src_ip` `src_port` `dst_ip` `dst_port` `proto` `trail_type` `trail` `trail_info` `reference` (e.g. `"2015-10-19 15:48:41.152513" beast 192.168.5.33 32985 8.8.8.8 53 UDP DNS 0000mps.webpreview.dsl.net malicious siteinspector.comodo.com`):

![Sample log](https://i.imgur.com/RycgVru.png)

### Server

Server configuration can be found inside the `maltrail.conf` section `[Server]`:

![Server's configuration](https://i.imgur.com/TiUpLX8.png)

*   `HTTP_ADDRESS`: Web server listening address (use `0.0.0.0` for all interfaces).
*   `HTTP_PORT`: Web server listening port (default: `8338`).
*   `USE_SSL`: If `true`, SSL/TLS is used for accessing the web server. In this case, `SSL_PEM` should point to the server's private/cert PEM file.
*   `USERS`: User configuration settings in the format `username:sha256(password):UID:filter_netmask(s)`.  `UID` is a unique user identifier (lower than 1000 recommended for administrative accounts).  `filter_netmask(s)` allows you to filter shown events depending on the user account.  Default entry:

![Configuration users](https://i.imgur.com/PYwsZkn.png)

*   `UDP_ADDRESS`: Server's log collecting listening address.
*   `UDP_PORT`: Listening port value.  Used in combination with `LOG_SERVER` for a distinct **Sensor** <-> **Server** architecture.
*   `FAIL2BAN_REGEX`:  Regular expression to use in `/fail2ban` web calls for extracting today's attacker source IPs (allows for IP blocking mechanisms like `fail2ban`, `iptables`, or `ipset`).
*   `BLACKLIST`: Build regular expressions to apply on one field (e.g. `src_ip`, `dst_port`). Syntax: `<field> <control> <regexp>`, where `control` can be `~` (matches) or `!~` (doesn't match). Use the `and` keyword to chain another rule. Example:

```
BLACKLIST_OUT
    src_ip !~ ^192.168. and dst_port ~ ^22$
    src_ip !~ ^192.168. and filter ~ scan
    src_ip !~ ^192.168. and filter ~ known attacker

BLACKLIST_IN
    src_ip ~ ^192.168. and filter ~ malware
```

As with the **Sensor**, running the **Server** for the first time, or after a long time, it will automatically update trails (if `USE_SERVER_UPDATE_TRAILS` is `true`). It stores log entries in the logging directory and provides the web reporting interface.

![Server run](https://i.imgur.com/GHdGPw7.png)

## User's Guide

### Reporting Interface

Upon accessing the **Server**'s reporting interface (via `HTTP_ADDRESS` and `HTTP_PORT`), the user is presented with an authentication dialog.  Users must enter the correct credentials set by the server administrator in `maltrail.conf` (default credentials: `admin:changeme!`):

![User login](https://i.imgur.com/WVpASAI.png)

Once logged in, you will be presented with the following reporting interface:

![Reporting interface](https://i.imgur.com/PZY8JEC.png)

*   **Timeline:**  Top section with a sliding timeline (activated by clicking the date label or the calendar icon). You can select logs for past events (mouse over the event to trigger the display of a tooltip with an approximate number of events for the current date). Dates are grouped by months, and a 4-month period of data is displayed. The provided slider (i.e. ![Timeline slider](https://i.imgur.com/SNGVSaP.png)) allows you to easily access events from previous months.
    ![Timeline](https://i.imgur.com/RnIROcn.png)
*   **Summary:** Middle section with a summary of displayed events.
    *   `Events`: Total number of events in a selected 24-hour period.
    *   `Sources`: Number of events per top sources in a stacked column chart.
    *   `Threats`: Percentage of top threats in a pie chart.
    *   `Trails`: Percentage of top trails in a pie chart.

![Summary](https://i.imgur.com/5NFbqCb.png)
*   **Event Table:** Bottom section with a condensed, paginated table of logged events. Each entry details a single threat (uniquely identified by `(src_ip, trail)` or `(dst_ip, trail)`).

![Single threat](https://i.imgur.com/IxPwKKZ.png)

*   `threat`: Unique ID and color derived from the threat's ID.
*   `sensor`:  Sensor name(s).
*   `events`: Total number of events for the current threat.
*   `severity`: Evaluated severity of the threat.
*   `first_seen`: Time of the first event.
*   `last_seen`: Time of the last event.
*   `sparkline`: Small sparkline graph of the threat's activity.
*   `src_ip`: Source IP(s).
*   `src_port`: Source port(s).
*   `dst_ip`: Destination IP(s).
*   `dst_port`: Destination port(s).
*   `proto`: Protocol(s).
*   `trail`: Blacklisted (or heuristic) entry that triggered the event(s).
*   `info`: More information about the threat/trail.
*   `reference`: Source of the blacklisted entry.
*   `tags`: User-defined tags.

*   **Tooltips**: Information tooltips with reverse DNS and WHOIS information are displayed when hovering over `src_ip` and `dst_ip`. The bubble icon displays condensed details, such as multiple ports, with a tooltip. Clicking the bubble opens a dialog with all stored items, ready to copy and paste. Hovering over the trail displays search engine results from [searX](https://searx.nixnet.services/).
*   **Tags:**  Use tags to categorize threats.

### Real-Life Cases

The following sections provide examples of how Maltrail can be used to identify real-world threats.

#### Mass Scans

Maltrail helps you identify scanning activity, often from services like Shodan or malicious actors.

![Shodan FileZilla results](https://i.imgur.com/nwOwLP9.png)

Reverse DNS and WHOIS lookup of the "attacker"'s address:

![Shodan 1](https://i.imgur.com/LQ6Vu00.png)

Search results from [searX](https://searx.nixnet.services/) by hovering over the `trail`:

![Shodan 2](https://i.imgur.com/vIzB8bA.png)

View scanned IP addresses:

![Shodan 3](https://i.imgur.com/EhAtXs7.png)

View ports scanned:

![Shodan 4](https://i.imgur.com/Wk8Xjhq.png)

#### Anonymous Attackers

Maltrail uses Tor exit node lists to identify attacks coming from the Tor network.

![Tor attacker](https://i.imgur.com/dXF8r2K.png)

#### Service Attackers

Maltrail also identifies attackers targeting specific services.

![RDP brute force](https://i.imgur.com/Oo2adCf.png)

Using filters, you can find specific attacks (e.g., SSH):

![SSH attackers filter](https://i.imgur.com/oCv42jd.png)

#### Malware

Maltrail helps you identify malware infections within your network.

![beebone malware](https://i.imgur.com/GBLWISo.png)

Identify DNS requests with DGA domains:

![necurs malware](https://i.imgur.com/8tWj2pm.png)

Detect file downloads from blacklisted URLs:

![malware download](https://i.imgur.com/g2NH7sT.png)

Filter for specific malware:

![ramnit malware](https://i.imgur.com/zcoPnZk.png)

Filter for all malware-related trails:

![malware filter](https://i.imgur.com/gVYAfSU.png)

#### Suspicious Domain Lookups

Maltrail uses lists of suspicious domains and heuristics to identify potentially malicious activity.

![cm DGA](https://i.imgur.com/JTGdtJ0.png)

Identify suspicious long domains:

![Suspicious long domains](https://i.imgur.com/EJOS5Qb.png)

Dynamic Domains:

![Suspicious dynamic domains](https://i.imgur.com/1WVLMf9.png)

Onion Domains:

![Suspicious onion](https://i.imgur.com/QdoAY0w.png)

Excessive no such domain name:

![Excessive no such domain name](https://i.imgur.com/KPwNOM8.png)

Flood Threats:

![Flood](https://i.imgur.com/ZtpMR3d.png)

#### Suspicious ipinfo requests

Malware often uses `ipinfo` services to identify victim IP addresses.

![suspicious ipinfo](https://i.imgur.com/3THOoWW.png)

Filter for potentially infected computers:

![ipinfo filter](https://i.imgur.com/6SMN0at.png)

#### Suspicious Direct File Downloads

Maltrail tracks suspicious direct file download attempts.

![Direct .exe download](https://i.imgur.com/jr5BS1h.png)

#### Suspicious HTTP Requests

Identify malicious requests, e.g., exploiting vulnerabilities.

![SQLi com_contenthistory](https://i.imgur.com/pZuGXpr.png)

Web application vulnerability scan marked as suspicious:

![Vulnerability scan](https://i.imgur.com/QzcaEsG.png)

Review the suspicious HTTP requests:

![Vulnerability scan requests](https://i.imgur.com/XY9K01o.png)

SQLmap scan requests:

![sqlmap scan requests](https://i.imgur.com/mHZmM7t.png)

#### Port Scanning

Maltrail alerts you about potential port scanning activities.

![nmap scan](https://i.imgur.com/VS7L2A3.png)

#### DNS Resource Exhaustion

Detect DDoS attacks that exhaust DNS server resources.

![DNS resource exhaustion](https://i.imgur.com/RujhnKW.png)

#### Data Leakage

Maltrail can detect suspicious data leakage activities.

![Data leakage](https://i.imgur.com/6zt2gXg.png)

#### False Positives

Maltrail may generate false positives, which administrators should review.

![Google false positive 1](https://i.imgur.com/HFvCNNK.png)

Example: Google server false positive with [searX](https://searx.nixnet.services/) search results:

![Google false positive 2](https://i.imgur.com/i3oydv6.png)

Suspicious domain false positive:

![Suspicious domain false positive](https://i.imgur.com/Msq8HgH.png)

## Best Practices

1.  **Installation**: Detailed installation steps for various systems are included in the [Quick Start](#quick-start) guide.
2.  **Configuration**: Configure the working and running environments as described in the [Administrator's Guide](#administrators-guide).
3.  **Automation**: Set up `crontab` jobs for automatic server startup, sensor startup, and trail updates as described in the [Best Practices](#best-practices) guide.
4.  **Systemd Services:** (Linux Only) Enable systemd services for automatic startup and management. See the [Best Practices](#best-practices) section.

## License

This software is provided under the MIT License. See the [LICENSE](https://github.com/stamparm/maltrail/blob/master/LICENSE) file for more information.

## Sponsors

*   [Sansec](https://sansec.io/) (2024-)
*   [Sansec](https://sansec.io/) (2020-2021)

## Developers

*   Miroslav Stampar ([@stamparm](https://github.com/stamparm))
*   Mikhail Kasimov ([@MikhailKasimov](https://github.com/MikhailKasimov))

## Presentations

*   47th TF-CSIRT Meeting, Prague (Czech Republic), 2016