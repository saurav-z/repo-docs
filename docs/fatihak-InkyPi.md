# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display"/>

**Transform your Raspberry Pi into a stylish and low-power information display with InkyPi, a customizable E-Ink solution.** ([Original Repository](https://github.com/fatihak/InkyPi))

## Key Features

*   **Paper-Like Display:** Enjoy crisp, easy-on-the-eyes visuals with no glare or backlighting.
*   **Web-Based Interface:** Easily configure and update your display from any device on your network.
*   **Minimalist Design:** Eliminate distractions with a display free of LEDs, noise, and notifications.
*   **Beginner-Friendly:** Simple installation and setup for makers of all levels.
*   **Open Source & Customizable:** Modify, expand, and create your own plugins to suit your needs.
*   **Scheduled Playlists:** Display different content at designated times with scheduled playlists.

## Plugins

InkyPi offers a growing library of plugins to display the information you care about:

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: Show daily comics and front pages of major newspapers.
*   Clock: Customizable clock faces.
*   AI Image/Text: Generate images and dynamic text using OpenAI's models.
*   Weather: Display current conditions and forecasts.
*   Calendar: Visualize your calendar from Google, Outlook, or Apple Calendar.

And more plugins are constantly being added!  For information on building custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - Recommended to use a 40-pin Pre-Soldered Header.
*   **MicroSD Card:** (min 8 GB) [Example Link](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**

    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. **Note: IT8951-based displays are not supported.** See [Waveshare e-Paper](#waveshare-display-support) compatibility for more information.
*   **Picture Frame or 3D Stand:**  See [community.md](./docs/community.md) for community-created designs and inspiration.

**Disclaimer:** Affiliate links are used; commissions earned help support the project.

## Installation

Get started with InkyPi in a few simple steps:

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```

2.  **Navigate to the Project Directory:**

    ```bash
    cd InkyPi
    ```

3.  **Run the Installation Script (with sudo):**

    *   **For Inky displays:**

        ```bash
        sudo bash install/install.sh
        ```

    *   **For Waveshare displays:**  Specify your display model.

        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        Example: `sudo bash install/install.sh -W epd7in3f`

After installation, the script will prompt you to reboot.  After rebooting, your display will show the InkyPi splash screen.

*   **Important Notes:**
    *   `sudo` privileges are needed for installation.  Start with a fresh Raspberry Pi OS installation to avoid potential conflicts.
    *   The install script automatically enables SPI and I2C interfaces.
    *   For detailed installation steps, including how to flash your microSD card, see [installation.md](./docs/installation.md) and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

Keep your InkyPi up to date with these commands:

1.  **Navigate to the Project Directory:**

    ```bash
    cd InkyPi
    ```

2.  **Fetch the Latest Changes:**

    ```bash
    git pull
    ```

3.  **Run the Update Script (with sudo):**

    ```bash
    sudo bash install/update.sh
    ```

This script applies updates and dependencies without requiring a complete reinstallation.

## Uninstalling InkyPi

To remove InkyPi from your Raspberry Pi, run:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is an active project with exciting plans for the future:

*   More Plugins
*   Modular Layouts
*   Button Support
*   Improved Web UI on Mobile

See the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for upcoming features and voting.

## Waveshare Display Support

InkyPi offers support for Waveshare e-Paper displays, which require model-specific drivers. **IT8951-based displays are not supported, and screens smaller than 4 inches are not recommended.** If your display model has a corresponding driver from the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it's likely compatible. Use the `-W` option during installation, specifying your display model (e.g., `-W epd7in3f`). The script will install the necessary drivers.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Support

*   Check the [troubleshooting guide](./docs/troubleshooting.md).
*   If you're still facing issues, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.
*   Pi Zero W users should review the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section.

## Sponsoring

Support the continued development of InkyPi:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and thanks to:

*   [PaperPi](https://github.com/txoof/PaperPi) - @txoof assisted with InkyPi's installation process.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - for modular plugins.
*   [PiInk](https://github.com/tlstommy/PiInk) - for the Flask web UI inspiration.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - for advanced power efficiency.