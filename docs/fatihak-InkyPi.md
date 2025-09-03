# InkyPi: Your Customizable E-Ink Display for Effortless Information

**[Visit the InkyPi GitHub Repository](https://github.com/fatihak/InkyPi)**

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="400"/>

Tired of screen glare and distracting notifications? InkyPi transforms a Raspberry Pi and E-Ink display into a sleek, low-power information hub, providing you with the content you care about, distraction-free.

## Key Features

*   **Paper-Like Aesthetics:** Enjoy crisp, minimalist visuals with no glare or backlight.
*   **Web-Based Configuration:** Easily set up and manage your display from any device on your network.
*   **Minimalist Design:** Eliminate LEDs, noise, and notifications, focusing solely on your chosen content.
*   **Beginner-Friendly Setup:** Simple installation and configuration make it perfect for makers of all levels.
*   **Open Source & Customizable:** Modify, expand, and create your own plugins with ease.
*   **Scheduled Playlists:** Display various content at different times with scheduled playlists.

## Core Plugins

InkyPi offers a variety of plugins to display useful information:

*   Image Upload
*   Daily Newspaper/Comic
*   Clock (Customizable)
*   AI Image/Text Generation (with OpenAI)
*   Weather Forecasts
*   Calendar Visualization (Google, Outlook, Apple Calendar)

And more plugins are on the way! For information on how to create custom plugins, check out [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

To get started with InkyPi, you will need the following components:

*   Raspberry Pi (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   MicroSD Card (min 8 GB) - [Example Link](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   **Pimoroni Inky Impression Displays:** (Links Below)
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT Displays:** (Links Below)
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:** (Links Below)
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibility for more information.
*   Picture Frame or 3D Stand
    *   See [community.md](./docs/community.md) for 3D models, custom builds, and other submissions from the community

**Disclaimer:** The links above are affiliate links.  I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps maintain and develop this project.

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
    *   **For Waveshare Displays:** (Replace `epd7in3f` with your specific model)

        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

    The script will prompt you to reboot your Raspberry Pi after installation. Your display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.  We recommend starting with a fresh Raspberry Pi OS installation.
*   The script automatically enables required SPI and I2C interfaces.
*   For more detailed installation instructions, see [installation.md](./docs/installation.md).

## Updating InkyPi

1.  **Navigate to the project directory:**

    ```bash
    cd InkyPi
    ```

2.  **Fetch the latest changes:**

    ```bash
    git pull
    ```

3.  **Run the update script:**

    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is actively evolving, with these features in development:

*   More Plugins
*   Modular layouts for flexible plugin combinations
*   Button support with customizable actions
*   Improved mobile Web UI

Check out the public [Trello Board](https://trello.com/b/SWJYWqe4/inkypi) to stay updated!

## Waveshare Display Support

InkyPi provides support for various Waveshare e-Paper displays.  Displays based on the IT8951 controller are not supported. The project is also not recommended for screens smaller than 4 inches due to the resolution.

**To ensure compatibility, verify your Waveshare display model has a driver available in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).**

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for more information.

## Attributions

This project utilizes fonts and icons with their own licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Visit the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues.  If you still need assistance, please create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

## Sponsoring

Support InkyPi's ongoing development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Sponsor on GitHub" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Check out these related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)