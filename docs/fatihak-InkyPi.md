# InkyPi: Your Customizable E-Ink Display for Effortless Content Display

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock">

**InkyPi transforms your Raspberry Pi into a sleek, low-power e-ink display, perfect for displaying the information you need without distractions.** [View the original repository on GitHub](https://github.com/fatihak/InkyPi).

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlight.
*   **Web-Based Configuration:** Easily set up and manage your display from any device on your network.
*   **Minimalist Design:** Eliminate distractions with no LEDs, noise, or notifications.
*   **Simple Setup:**  Designed for both beginners and experienced makers.
*   **Open Source:** Modify, customize, and create your own plugins.
*   **Scheduled Playlists:** Display different plugins at designated times.

## Plugins

InkyPi offers a variety of plugins to display the information you care about, with more in development!

*   Image Upload
*   Daily Newspaper/Comic
*   Clock (Customizable)
*   AI Image/Text Generation
*   Weather Forecasts
*   Calendar Visualization

For information on building custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB) (e.g., [Amazon Link](https://amzn.to/3G3Tq9W))
*   **E-Ink Display:**

    *   **Inky Impression by Pimoroni:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color:  [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   More models at: [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or their [Amazon store](https://amzn.to/3HPRTEZ).  Note that IT8951-based displays are **not** supported. See the [Waveshare Display Support](#waveshare-display-support) section below.
*   **Picture Frame or 3D Stand:** See [community.md](./docs/community.md) for community-created designs.

**Affiliate Disclosure:**  Some links are affiliate links, and I may earn a commission from qualifying purchases, at no extra cost to you. This helps support the project.

## Installation

Follow these steps to install InkyPi:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script (with `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`:  Specify this option *only* if using a Waveshare display (e.g., `-W epd7in3f`).
    *   For Inky displays:
        ```bash
        sudo bash install/install.sh
        ```
    *   For Waveshare displays:
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, reboot your Raspberry Pi.  The display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.  Start with a fresh Raspberry Pi OS installation to avoid conflicts.
*   SPI and I2C interfaces are automatically enabled.
*   Refer to [installation.md](./docs/installation.md) for more detailed instructions, including imaging your microSD card.  Also, watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

To update your InkyPi installation:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

To uninstall InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

Explore upcoming features and vote on what you want to see next!

*   More plugins!
*   Modular layouts
*   Button support with custom actions
*   Improved Web UI on mobile devices

Check out the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays.

**Compatibility:**  If your Waveshare display has a driver in their [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it's likely compatible.

**Important Notes:**

*   **IT8951 Controller Not Supported.**
*   **Displays Smaller Than 4 Inches Are Not Recommended.**
*   Use the `-W` option during installation, specifying your display model (without the `.py` extension).  The script will install the necessary drivers.

## License

Distributed under the GPL 3.0 License ([LICENSE](./LICENSE)).

This project uses fonts and icons with their own licensing; see [Attribution](./docs/attribution.md).

## Troubleshooting

See the [troubleshooting guide](./docs/troubleshooting.md).  If you still need help, open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W Users:**  See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section for details.

## Sponsoring

Support the ongoing development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these similar projects:

*   [PaperPi](https://github.com/txoof/PaperPi) - Supports Waveshare devices.  Shoutout to @txoof for installation assistance.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspiration for the Flask web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - Weather dashboard with advanced power efficiency.