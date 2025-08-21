# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[View the original repository](https://github.com/fatihak/InkyPi)

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display">

Tired of distracting screens? **InkyPi transforms your Raspberry Pi into a beautiful, low-power e-ink display that shows you exactly what you need, when you need it.**

## Key Features

*   **Eye-Friendly Display:** Enjoy a paper-like aesthetic with no glare or backlighting.
*   **Web-Based Control:** Easily configure and update your display from any device on your network.
*   **Minimalist Experience:** Eliminate distractions with a display free of LEDs, noise, and notifications.
*   **Simple Setup:** Get started quickly with easy installation and configuration, perfect for beginners.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins.
*   **Scheduled Playlists:** Display different content at set times.

## Plugins

InkyPi offers a range of plugins to display the information you care about, with more on the way!

*   **Image Upload:** Display any image you upload.
*   **Daily Newspaper/Comic:** Read your favorite comics and news headlines.
*   **Clock:** Choose from customizable clock faces.
*   **AI Image/Text:** Generate images and text using AI models.
*   **Weather:** View current conditions and forecasts.
*   **Calendar:** Visualize your appointments from various calendar services.

For information on creating custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40 pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression Displays:**
        *   [13.3 Inch](https://collabs.shop/q2jmza)
        *   [7.3 Inch](https://collabs.shop/q2jmza)
        *   [5.7 Inch](https://collabs.shop/ns6m6m)
        *   [4 Inch](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT Displays:**
        *   [4.2 Inch](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) | [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) | [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See the [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.  **Note:** IT8951 based displays are not supported. See [Waveshare Display Support](#waveshare-display-support) for more information.

*   **Picture Frame or 3D Stand:**
    *   See [community.md](./docs/community.md) for community-contributed designs and builds.

**Affiliate Disclosure:** The links provided are affiliate links. Commissions earned help support project development, at no extra cost to you.

## Installation

Get InkyPi up and running with these easy steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script (using `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   **`-W <waveshare device model>`:**  *Use ONLY for Waveshare displays.* Specify your display model (e.g., `-W epd7in3f`).
    *   **For Inky displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare displays:**
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, reboot your Raspberry Pi.  The display will show the InkyPi splash screen.  Refer to [installation.md](./docs/installation.md) and the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4) for detailed instructions.

*   **Important Notes:** The install script requires `sudo` privileges. It's best to start with a fresh Raspberry Pi OS installation. The script will automatically enable SPI and I2C.

## Updating InkyPi

Keep your InkyPi updated:

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch the latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script (using `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling InkyPi

Remove InkyPi with a single command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is constantly improving!  Upcoming features include:

*   More plugins!
*   Modular layouts
*   Button support
*   Improved Web UI (mobile-friendly)

See the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for the latest plans and to vote on features.

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays. To install the correct drivers, follow the instructions above, and use the `-W` option to specify your display model when running the installation script.

**Note:**

*   Displays based on the IT8951 controller are not supported.
*   Screens smaller than 4 inches are not recommended due to resolution limitations.
*   Ensure your display model has a corresponding driver in the Waveshare [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE).
This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md).

## Troubleshooting & Support

Check the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues. If you need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

If you're using a Pi Zero W, be aware of potential issues during the installation.  See the troubleshooting guide for details.

## Sponsoring

Support InkyPi's ongoing development!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and/or incorporating elements from:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi\_weather\_display](https://github.com/sjnims/rpi_weather_display)