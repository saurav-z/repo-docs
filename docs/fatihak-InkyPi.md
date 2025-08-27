# InkyPi: Your Customizable E-Ink Display for a Distraction-Free Life

[View the original repository on GitHub](https://github.com/fatihak/InkyPi)

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" />

Tired of notifications and screen glare? **InkyPi transforms a Raspberry Pi and E-Ink display into a beautiful, low-power, and customizable information hub.**

## Key Features:

*   **E-Ink Clarity:** Enjoy a paper-like aesthetic with no glare or backlighting for a comfortable viewing experience.
*   **Web-Based Control:** Easily configure and update your display from any device on your network using a simple web interface.
*   **Distraction-Free Design:** Focus on what matters with a display designed to minimize distractions.
*   **Simple Setup:** Get started quickly with easy installation and configuration, perfect for beginners.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins to display exactly what you need.
*   **Scheduled Playlists:** Set up playlists to cycle through different plugins at designated times.

## Plugins:

InkyPi comes with a growing selection of plugins to display the information you need:

*   Image Upload
*   Daily Newspaper/Comic
*   Clock (Customizable)
*   AI Image/Text Generation (via OpenAI)
*   Weather Forecasts
*   Calendar Visualization

More plugins are constantly being added!  Check out the [Building InkyPi Plugins](./docs/building_plugins.md) documentation for details on creating your own.

## Hardware Requirements:

*   Raspberry Pi (4, 3, or Zero 2 W - 40 pin Pre Soldered Header Recommended)
*   MicroSD Card (min 8 GB) like [this one](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   Inky Impression by Pimoroni (various sizes available)
    *   Inky wHAT by Pimoroni (4.2 Inch)
    *   Waveshare e-Paper Displays (various sizes and colors)

    *   **Note:** The provided links are affiliate links, and I may earn a commission from qualifying purchases.
    See [community.md](./docs/community.md) for 3D models, custom builds, and other submissions from the community

## Installation:

1.  Clone the repository:
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
3.  Run the installation script:
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **For Inky Displays:**  Use the command above without the `-W` option.
    *   **For Waveshare Displays:**  Specify your display model (e.g., `-W epd7in3f`). See the [Waveshare Display Support](#waveshare-display-support) section below.

    Refer to [installation.md](./docs/installation.md) or the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4) for detailed instructions.

## Updating InkyPi:

1.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
2.  Fetch the latest changes:
    ```bash
    git pull
    ```
3.  Run the update script:
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap:

*   Plugin Expansion
*   Modular Layouts
*   Button Support
*   Improved Mobile Web UI

Stay updated on the latest features and vote for your favorites on the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support:

InkyPi supports a variety of Waveshare e-Paper displays.  When using Waveshare displays, **specify the model** during installation using the `-W` option.

*   **Note:** Displays based on the IT8951 controller are *not* supported, and screens smaller than 4 inches are *not* recommended. Refer to the [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) for the full range of compatible models

## License:

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for more information.

This project uses fonts and icons with separate licensing. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Issues:

Check the [troubleshooting guide](./docs/troubleshooting.md).  For unresolved issues, open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.
*   **Pi Zero W Users:** Be aware of [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation)

## Sponsoring:

Support InkyPi's development!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements:

Inspired by and with special thanks to:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)