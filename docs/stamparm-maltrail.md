[![Python 2.6|2.7|3.x](https://img.shields.io/badge/python-2.6|2.7|3.x-yellow.svg)](https://www.python.org/) [![License](https://img.shields.io/badge/license-MIT-red.svg)](https://github.com/stamparm/maltrail#license) [![Malware families](https://img.shields.io/badge/malware_families-1494-orange.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Malware sinkholes](https://img.shields.io/badge/malware_sinkholes-1354-green.svg)](https://github.com/stamparm/maltrail/tree/master/trails/static/malware) [![Twitter](https://img.shields.io/badge/twitter-@maltrail-blue.svg)](https://twitter.com/maltrail)

## Maltrail: Detect and Mitigate Malicious Network Traffic

Maltrail is a powerful, open-source malicious traffic detection system that helps you identify and respond to threats on your network.  [View the original repository](https://github.com/stamparm/maltrail).

**Key Features:**

*   **Real-time Threat Detection:** Monitors network traffic for malicious indicators.
*   **Extensive Blacklist Integration:** Uses publicly available blacklists, static trails, and custom lists.
*   **Heuristic Analysis:** Employs advanced mechanisms to identify unknown threats.
*   **Flexible Architecture:** Based on a Traffic -> Sensor <-> Server <-> Client architecture, enabling standalone or centralized deployment.
*   **Reporting Interface:** Web-based interface for threat visualization and analysis.
*   **Integration:** Support for integration with third-party tools like Wazuh, Splunk, and Palo Alto Networks Cortex XSOAR.
*   **Multiple Protocol Support:** Analyze DNS, HTTP, and other protocols.
*   **OS Support:** Supports Python 2.6, 2.7, and 3.x on \*nix/BSD systems.
*   **Docker Support:** Easy deployment via Docker containers.

**Table of Contents**

*   [Introduction](#introduction)
*   [Architecture](#architecture)
*   [Demo Pages](#demo-pages)
*   [Quick Start](#quick-start)
*   [Administrator's Guide](#administrators-guide)
    *   [Sensor](#sensor)
    *   [Server](#server)
*   [User's Guide](#users-guide)
    *   [Reporting Interface](#reporting-interface)
*   [Real-life Cases](#real-life-cases)
*   [Best Practices](#best-practices)
*   [License](#license)
*   [Sponsors](#sponsors)
*   [Developers](#developers)
*   [Presentations](#presentations)
*   [Publications](#publications)
*   [Blacklist](#blacklist)
*   [Thank You](#thank-you)
*   [Third-party Integrations](#third-party-integrations)

## Introduction

**Maltrail** is a malicious traffic detection system designed to identify and alert on suspicious network activity. It leverages a combination of publicly available blacklists, static trails derived from AV reports, and user-defined lists to detect threats.  Trails can be anything from domain names (e.g., `zvpprsensinaix.com` for the Banjori malware), URLs (e.g., `hXXp://109.162.38.120/harsh02.exe` for known malicious executables), IP addresses (e.g., `185.130.5.231` for known attackers), or HTTP User-Agent header values (e.g., `sqlmap` for automatic SQL injection tools). In addition, Maltrail incorporates heuristic mechanisms to identify novel threats.

![Reporting tool](https://i.imgur.com/Sd9eqoa.png)

Maltrail utilizes various (black)lists (i.e. feeds):

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

Maltrail also includes trails for the following malicious entities:

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

Maltrail's architecture consists of the **Traffic** -> **Sensor** <-> **Server** <-> **Client** components.  The **Sensor**(s) is a standalone component that runs on the monitoring node (e.g., a Linux platform connected to a SPAN/mirroring port or placed inline on a Linux bridge). It monitors network **Traffic** for blacklisted items (domain names, URLs, IPs). Upon a match, it sends event details to the (central) **Server**, where they are stored in a logging directory.

![Architecture diagram](https://i.imgur.com/2IP9Mh2.png)

The **Server** stores event details and provides backend support for the reporting web application. By default, the sensor and server run on the same machine. The client-side reporting (the **Client**) uses a "Fat client" approach, where the data is processed inside the client's web browser. This allows for the presentation of a practically unlimited number of events.  The **Server** component can be skipped and the standalone **Sensor** can be used.

## Demo Pages

Explore the functionality of Maltrail with demo pages featuring real-life threat data, found [here](https://maltraildemo.github.io/).

## Quick Start

Here are the commands to get the Maltrail **Sensor** up and running quickly:

*   **Ubuntu/Debian**

    ```sh
    sudo apt-get install git python3 python3-dev python3-pip python-is-python3 libpcap-dev build-essential procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```

*   **SUSE/openSUSE**

    ```sh
    sudo zypper install gcc gcc-c++ git libpcap-devel python3-devel python3-pip procps schedtool
    sudo pip3 install pcapy-ng
    git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    sudo python3 sensor.py
    ```
    
    Don't forget to put interfaces in promiscuous mode as needed: 

    ```sh
    for dev in $(ifconfig | grep mtu | grep -Eo '^\w+'); do ifconfig $dev promisc; done
    ```

    ![Sensor](https://i.imgur.com/E9tt2ek.png)

    To start the (optional) **Server** on same machine, open a new terminal and execute the following:

    ```sh
    [[ -d maltrail ]] || git clone --depth 1 https://github.com/stamparm/maltrail.git
    cd maltrail
    python server.py
    ```

    ![Server](https://i.imgur.com/loGW6GA.png)

*   **Docker**

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

## Administrator's Guide

### Sensor

Sensor's configuration is managed in the `maltrail.conf` file under the `[Sensor]` section:

![Sensor's configuration](https://i.imgur.com/8yZKH14.png)

Key settings:

*   `USE_MULTIPROCESSING`: Enables multi-core processing.
*   `USE_FEED_UPDATES`: Controls trail updates from feeds.
*   `UPDATE_PERIOD`: Sets the time (in seconds) between automatic trail updates.
*   `CUSTOM_TRAILS_DIR`: Sets the directory containing custom trail files.
*   `USE_HEURISTICS`: Enables heuristic mechanisms for detecting suspicious traffic.
*   `CAPTURE_BUFFER`:  Sets the memory buffer size for multiprocessing mode.
*   `MONITOR_INTERFACE`: Specifies the network interface to monitor.  Use `any` to monitor all interfaces.
*   `CAPTURE_FILTER`: Uses a tcpdump-compatible filter to focus on specific traffic.
*   `SENSOR_NAME`: Sets the name of the sensor to distinguish it in logs.
*   `LOG_SERVER`: Configures remote logging to a server.
*   `LOG_DIR`: Specifies the directory for local logging.
*   `UPDATE_SERVER`: Specify a URL for downloading updated trail definitions
*   `SYSLOG_SERVER` and/or `LOGSTASH_SERVER`:  Allows the forwarding of sensor events to a Syslog or Logstash server.

### Server

Server configuration can be found inside the `maltrail.conf` section `[Server]`:

![Server's configuration](https://i.imgur.com/TiUpLX8.png)

Key settings:

*   `HTTP_ADDRESS`: Sets the web server's listening address. Use `0.0.0.0` for all interfaces.
*   `HTTP_PORT`: Defines the web server's listening port.
*   `USE_SSL`: Enables SSL/TLS for secure access, `SSL_PEM` points to the PEM file.
*   `USERS`: Contains user configuration settings (username:password:UID:filter_netmask(s)).
*   `UDP_ADDRESS`: Configures the server's log collection listening address (for remote sensors).
*   `UDP_PORT`: Sets the UDP port for log collection.
*   `FAIL2BAN_REGEX`: Configure regex to be used for `/fail2ban` web calls.
*   `BLACKLIST`: Enables regular expressions that can be used to apply on one field and create blacklists (e.g. from web calls).

## User's Guide

### Reporting Interface

The reporting interface provides a web-based view of detected threats.

1.  **Authentication:** Log in using credentials defined in `maltrail.conf`.
    ![User login](https://i.imgur.com/WVpASAI.png)
2.  **Dashboard:** The main dashboard provides an overview of threats.
    ![Reporting interface](https://i.imgur.com/PZY8JEC.png)
3.  **Timeline:** Select events from a specific date.
    ![Timeline](https://i.imgur.com/RnIROcn.png)
4.  **Summary:** Visualize event summaries, top sources, threats, and trails.
    ![Summary](https://i.imgur.com/5NFbqCb.png)
5.  **Event Table:** View a paginated table of logged events.
    ![Single threat](https://i.imgur.com/IxPwKKZ.png)

*   Hovering over `src_ip` and `dst_ip` displays reverse DNS and WHOIS information.
*   Bubble icons indicate condensed event details.
*   Hovering over a threat's `trail` provides results from [searX](https://searx.nixnet.services/).
*   Use tags to organize and categorize threats.

### Real-life Cases

The following sections will present several "usual suspects" and scenarios that will be described through the real-life cases.

#### Mass scans
Examples from Shodan:

![Shodan FileZilla results](https://i.imgur.com/nwOwLP9.png)
![Shodan 1](https://i.imgur.com/LQ6Vu00.png)
![Shodan 2](https://i.imgur.com/vIzB8bA.png)
![Shodan 3](https://i.imgur.com/EhAtXs7.png)
![Shodan 4](https://i.imgur.com/Wk8Xjhq.png)

#### Anonymous attackers
Example: using Tor:

![Tor attacker](https://i.imgur.com/dXF8r2K.png)

#### Service attackers
Examples of brute-force attacks:

![RDP brute force](https://i.imgur.com/Oo2adCf.png)
![SSH attackers filter](https://i.imgur.com/oCv42jd.png)

#### Malware
Examples of malware detections:

![beebone malware](https://i.imgur.com/GBLWISo.png)
![necurs malware](https://i.imgur.com/8tWj2pm.png)
![malware download](https://i.imgur.com/g2NH7sT.png)
![ramnit malware](https://i.imgur.com/zcoPnZk.png)
![malware filter](https://i.imgur.com/gVYAfSU.png)

#### Suspicious domain lookups
Examples of DGA and other suspicious domain activity:

![cm DGA](https://i.imgur.com/JTGdtJ0.png)
![Suspicious long domains](https://i.imgur.com/EJOS5Qb.png)
![Suspicious dynamic domains](https://i.imgur.com/1WVLMf9.png)
![Suspicious onion](https://i.imgur.com/QdoAY0w.png)
![Excessive no such domain name](https://i.imgur.com/KPwNOM8.png)
![Flood](https://i.imgur.com/ZtpMR3d.png)

#### Suspicious ipinfo requests
Examples:

![suspicious ipinfo](https://i.imgur.com/3THOoWW.png)
![ipinfo filter](https://i.imgur.com/6SMN0at.png)

#### Suspicious direct file downloads
Examples:

![Direct .exe download](https://i.imgur.com/jr5BS1h.png)

#### Suspicious HTTP requests
Examples:

![SQLi com_contenthistory](https://i.imgur.com/pZuGXpr.png)
![Vulnerability scan](https://i.imgur.com/QzcaEsG.png)
![sqlmap scan requests](https://i.imgur.com/mHZmM7t.png)

#### Port scanning
Examples:

![nmap scan](https://i.imgur.com/VS7L2A3.png)

#### DNS resource exhaustion
Examples:

![DNS resource exhaustion](https://i.imgur.com/RujhnKW.png)

#### Data leakage
Examples:

![Data leakage](https://i.imgur.com/6zt2gXg.png)

#### False positives
Examples:

![Google false positive 1](https://i.imgur.com/HFvCNNK.png)
![Google false positive 2](https://i.imgur.com/i3oydv6.png)
![Suspicious domain false positive](https://i.imgur.com/Msq8HgH.png)
![Suspicious .ws](https://i.imgur.com/bOLmXUE.png)

## Best Practices

Follow these steps for best results:

1.  Install Maltrail.
2.  Configure the working and running environment.

    ```sh
    sudo mkdir -p /var/log/maltrail
    sudo mkdir -p /etc/maltrail
    sudo cp /opt/maltrail/maltrail.conf /etc/maltrail
    sudo nano /etc/maltrail/maltrail.conf
    ```
3.  Set up autostart using `crontab`.
4.  Enable as systemd services (Linux only).

## License

Maltrail is licensed under the MIT License. See the [LICENSE](https://github.com/stamparm/maltrail/blob/master/LICENSE) file for details.

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

Maltrail provides a daily updated blacklist of malware-related domains, found [here](https://raw.githubusercontent.com/stamparm/aux/master/maltrail-malware-domains.txt).

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

## Third-party Integrations

*   [FreeBSD Port](https://www.freshports.org/security/maltrail)
*   [OPNSense Gateway Plugin](https://github.com/opnsense/plugins/pull/1257)
*   [D4 Project](https://www.d4-project.org/2019/09/25/maltrail-integration.html)
*   [BlackArch Linux](https://github.com/BlackArch/blackarch/blob/master/packages/maltrail/PKGBUILD)
*   [Validin LLC](https://twitter.com/ValidinLLC/status/1719666086390517762)
*   [Maltrail Add-on for Splunk](https://splunkbase.splunk.com/app/7211)
*   [Maltrail decoder and rules for Wazuh](https://github.com/MikhailKasimov/maltrail-wazuh-decoder-and-rules)
*   [GScan](https://github.com/grayddq/GScan) <sup>1</sup>
*   [MalwareWorld](https://www.malwareworld.com/) <sup>1</sup>
*   [oisd | domain blocklist](https://oisd.nl/?p=inc) <sup>1</sup>
*   [NextDNS](https://github.com/nextdns/metadata/blob/e0c9c7e908f5d10823b517ad230df214a7251b13/security/threat-intelligence-feeds.json) <sup>1</sup>
*   [NoTracking](https://github.com/notracking/hosts-blocklists/blob/master/SOURCES.md) <sup>1</sup>
*   [OWASP Mobile Audit](https://github.com/mpast/mobileAudit#environment-variables) <sup>1</sup>
*   [Mobile-Security-Framework-MobSF](https://github.com/MobSF/Mobile-Security-Framework-MobSF/commit/12b07370674238fa4281fc7989b34decc2e08876) <sup>1</sup>
*   [pfBlockerNG-devel](https://github.com/pfsense/FreeBSD-ports/blob/devel/net/pfSense-pkg-pfBlockerNG-devel/files/usr/local/www/pfblockerng/pfblockerng_feeds.json) <sup>1</sup>
*   [Sansec eComscan](https://sansec.io/kb/about-ecomscan/ecomscan-license)<sup>1</sup>
*   [Palo Alto Networks Cortex XSOAR](https://xsoar.pan.dev/docs/reference/integrations/github-maltrail-feed)<sup>2</sup>

<sup>1</sup> Using (only) trails

<sup>2</sup> Connector to trails (only)