# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display" />

**Transform your Raspberry Pi into a stylish, low-power information hub with InkyPi, an open-source project for displaying dynamic content on E-Ink displays.**  [View the original repository on GitHub](https://github.com/fatihak/InkyPi).

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting on your E-Ink display.
*   **Web-Based Interface:** Effortlessly configure and update your display from any device on your network.
*   **Minimalist Design:** Eliminate distractions with a display free of LEDs, noise, and unnecessary notifications.
*   **Easy Setup:** Simple installation and configuration, perfect for beginners and experienced makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins with the open-source nature of InkyPi.
*   **Scheduled Content:** Set up playlists to display different content throughout the day.
*   **Multiple Plugin Support:** Supports a variety of plugins for displaying different types of information.

## Plugins

InkyPi offers a growing collection of plugins to display various types of information:

*   Image Upload
*   Daily Newspaper/Comic
*   Clock (Customizable)
*   AI Image/Text Generation (powered by OpenAI)
*   Weather Forecasts
*   Calendar Visualization

[Explore the documentation to create your own custom plugins.](./docs/building_plugins.md)

## Hardware Requirements

To get started, you'll need the following hardware:

*   Raspberry Pi (4, 3, or Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   MicroSD Card (minimum 8 GB), such as [this one](https://amzn.to/3G3Tq9W).
*   E-Ink Display (Choose from the following):
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
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.  Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   Picture Frame or 3D Stand (optional)
    *   See [community.md](./docs/community.md) for 3D models and community contributions.

**Affiliate Disclosure:** The links above are affiliate links.  I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps maintain and develop this project.

## Installation

Follow these steps to install InkyPi:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script (with `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`:  **ONLY** specify this if installing for a Waveshare display. Replace `<waveshare device model>` with your display's model (e.g., `epd7in3f`).

    **Examples:**

    *   For Inky displays: `sudo bash install/install.sh`
    *   For Waveshare displays: `sudo bash install/install.sh -W epd7in3f`

After installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

**Important Notes:**

*   `sudo` privileges are required. It is recommended to start with a fresh Raspberry Pi OS installation.
*   The script will automatically enable necessary SPI and I2C interfaces.

[Refer to installation.md](./docs/installation.md) for detailed instructions, including how to image your microSD card.  You can also watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

Keep your InkyPi up-to-date with these steps:

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

To completely remove InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is constantly evolving! Future development includes:

*   More Plugins!
*   Modular layouts
*   Button support
*   Improved Web UI for mobile devices

[Visit the public Trello board](https://trello.com/b/SWJYWqe4/inkypi) to see upcoming features and vote on your favorites!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays, which have different driver requirements than the Inky Impression displays.  **Displays based on the IT8951 controller are not supported**, and **screens smaller than 4 inches are not recommended** due to limited resolution.  Use the `-W` option with the installation script to install the appropriate Waveshare driver for your model.

## License

Distributed under the GPL 3.0 License.  See [LICENSE](./LICENSE) for details.

This project incorporates fonts and icons with their own licensing requirements. See [Attribution](./docs/attribution.md) for more information.

## Troubleshooting

Check out the [troubleshooting guide](./docs/troubleshooting.md). If you still need assistance, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known issues with Pi Zero W:** See the [troubleshooting guide](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) for details.

## Sponsoring

Support the continued development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Sponsor on GitHub" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Similar projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)