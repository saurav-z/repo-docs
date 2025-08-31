# InkyPi: Your Customizable E-Ink Display

**Transform your Raspberry Pi into a low-power, distraction-free information hub with InkyPi, an open-source project that lets you display the content you care about on a crisp, paper-like E-Ink display.**

[View the original repository on GitHub](https://github.com/fatihak/InkyPi)

## Key Features

*   **E-Ink Display:** Experience a natural, paper-like aesthetic with no glare or backlight, perfect for eye comfort.
*   **Web Interface:** Effortlessly configure and update your display from any device on your network.
*   **Minimalist Design:** Minimize distractions with no LEDs, noise, or notifications; just the information you need.
*   **Easy Setup:** Simple installation and configuration, ideal for beginners and experienced makers.
*   **Open Source & Customizable:** Modify, customize, and create your own plugins to fit your specific needs.
*   **Scheduled Playlists:** Set up playlists to display different plugins at designated times.

## Plugins

InkyPi offers a variety of plugins to display diverse content:

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: Show daily comics and front pages of major newspapers from around the world.
*   Clock: Customizable clock faces.
*   AI Image/Text: Generate images and dynamic text from prompts using OpenAI's models.
*   Weather: Current weather conditions and forecasts.
*   Calendar: Visualize your calendar from Google, Outlook, or Apple Calendar.

Explore [Building InkyPi Plugins](./docs/building_plugins.md) to create custom plugins.

## Hardware

*   Raspberry Pi (4 | 3 | Zero 2 W) - Recommended with a 40-pin Pre-Soldered Header
*   MicroSD Card (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   **Inky Impression by Pimoroni**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.

*   Picture Frame or 3D Stand - See [community.md](./docs/community.md) for community contributions.

**Disclosure:** *This README contains affiliate links. I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps maintain and develop this project.*

## Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  Navigate to the project directory:

    ```bash
    cd InkyPi
    ```
3.  Run the installation script with sudo:

    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **-W <waveshare device model>** - Specify this parameter **ONLY** for Waveshare displays.

    Examples:

    *   For Inky displays: `sudo bash install/install.sh`
    *   For Waveshare displays: `sudo bash install/install.sh -W epd7in3f`

    After installation, reboot your Raspberry Pi. The display will show the InkyPi splash screen.

    *   **Note:** Requires sudo privileges and a fresh Raspberry Pi OS installation is recommended.
    *   For more detailed instructions, see [installation.md](./docs/installation.md) or watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

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

## Uninstall

To uninstall InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   Plugins, plugins, plugins
*   Modular layouts to mix and match plugins
*   Support for buttons with customizable action bindings
*   Improved Web UI on mobile devices

Check out the [public trello board](https://trello.com/b/SWJYWqe4/inkypi) to explore upcoming features and vote on what you'd like to see next!

## Waveshare Display Support

This project supports Waveshare e-Paper displays. **IT8951 controller based displays are not supported**, and **screens smaller than 4 inches are not recommended** due to limited resolution. If your display model has a driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it's likely compatible. Use the `-W` option during installation.

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Issues & Support

*   Check the [troubleshooting guide](./docs/troubleshooting.md).
*   Create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.
*   **Pi Zero W Note:** See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section in the troubleshooting guide.

## Sponsoring

Support InkyPi's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)