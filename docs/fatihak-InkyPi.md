# InkyPi: Your Customizable E-Ink Display for a Minimalist Lifestyle

[![InkyPi Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi and E-Ink display into a sleek, low-power dashboard with InkyPi, offering a distraction-free way to view the information you need.**

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting.
*   **Web-Based Control:** Easily configure and update your display from any device on your network.
*   **Minimalist Design:** Eliminate distractions with a display free of LEDs, noise, and notifications.
*   **Easy Setup:** Get up and running quickly with a straightforward installation process perfect for all skill levels.
*   **Open Source:** Customize, modify, and build your own plugins.
*   **Scheduled Playlists:** Display different content at specific times.
*   **Plugin Variety:** Ready-made plugins including: Image Upload, News & Comics, Clock, AI-Generated Content, Weather, Calendar, and more!

[Back to the original repository](https://github.com/fatihak/InkyPi)

## Hardware

*   **Raspberry Pi** (4, 3, or Zero 2 W)
    *   Recommended to get 40 pin Pre Soldered Header
*   **MicroSD Card** (min 8 GB)
*   **E-Ink Display:**
    *   Inky Impression by Pimoroni (13.3", 7.3", 5.7", 4")
    *   Inky wHAT by Pimoroni (4.2")
    *   Waveshare e-Paper Displays (various sizes and colors)
        *   Spectra 6 (E6) Full Color (4", 7.3", 13.3")
        *   Black and White (7.5", 13.3")
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   **Picture Frame or 3D Stand**
    *   See [community.md](./docs/community.md) for 3D models, custom builds, and other community contributions.

**Note:** Affiliate links are included, which may provide a commission to support the project at no extra cost to you.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script:**

    *   **For Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:** (Specify your display model)
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        *Example: `sudo bash install/install.sh -W epd7in3f`*

    After installation, reboot your Raspberry Pi.

    For detailed instructions, refer to [installation.md](./docs/installation.md) and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To remove InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More Plugins
*   Modular layouts
*   Button support
*   Improved Web UI on mobile

Explore and vote on upcoming features on the [public Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays. Use the `-W` option during installation and specify your display model.

**Important Notes:**

*   IT8951 controller-based displays are not supported.
*   Screens smaller than 4 inches are not recommended.
*   Ensure your display model has a corresponding driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

## Attributions

This project includes fonts and icons with separate licensing. See [Attribution](./docs/attribution.md) for details.

## Issues

Check the [troubleshooting guide](./docs/troubleshooting.md) or open a [GitHub Issue](https://github.com/fatihak/InkyPi/issues).

## Sponsoring

Support the development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Explore related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)
```
Key improvements and SEO-optimized elements:

*   **Clear and concise title:** "InkyPi: Your Customizable E-Ink Display for a Minimalist Lifestyle" includes a relevant keyword and highlights the value proposition.
*   **One-sentence hook:**  The introductory sentence immediately grabs the reader's attention and explains the core benefit.
*   **Descriptive headings:**  Use of clear, keyword-rich headings like "Key Features", "Hardware", "Installation", etc. improve readability and SEO.
*   **Bulleted lists:** The use of bulleted lists is a standard practice for quickly conveying information.
*   **Keyword optimization:**  The README strategically uses relevant keywords like "E-Ink display," "Raspberry Pi," "customizable," "minimalist," "plugins," "web interface," and names of specific features to improve search visibility.
*   **Clear Calls to Action:** The inclusion of links to the original repo.
*   **Concise summaries:** The descriptions are succinct, avoiding unnecessary jargon.
*   **Emphasis on benefits:** The README highlights the advantages of using InkyPi (e.g., eye-friendly, distraction-free, easy to set up).
*   **Improved Structure:**  The organization of information is logical and easy to follow.
*   **Internal links**: The document links to helpful internal documentation pages like the `installation.md` and the `troubleshooting.md`.
*   **Affiliate Link Disclosure**: Maintains transparency.