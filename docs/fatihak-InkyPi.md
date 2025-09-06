# InkyPi: Your Customizable E-Ink Display for a Distraction-Free Life

[![InkyPi E-Ink Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**InkyPi** is an open-source, customizable E-Ink display powered by a Raspberry Pi, designed to effortlessly display the information you care about with a beautiful, paper-like aesthetic. [Explore InkyPi on GitHub](https://github.com/fatihak/InkyPi)!

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, minimalist visuals with no glare or backlighting.
*   **Web-Based Interface:** Configure and update your display from any device on your network.
*   **Distraction-Free Experience:** Minimize distractions with no LEDs, noise, or notifications.
*   **Easy Setup & Configuration:** Perfect for beginners and experienced makers alike.
*   **Open Source:** Modify, customize, and create your own plugins.
*   **Scheduled Playlists:** Display different content at designated times.

## Plugins: Display What Matters Most

InkyPi comes with a growing library of plugins to display a wide range of information:

*   Image Upload
*   Daily Newspaper/Comic
*   Clock
*   AI Image/Text Generation
*   Weather Forecasts
*   Calendar Visualization
*   And more coming soon!

For information on building custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   Raspberry Pi (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   MicroSD Card (min 8 GB) like [this one](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   **Inky Impression by Pimoroni:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Inky wHAT by Pimoroni:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) / [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) / [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) / [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.

*   Picture Frame or 3D Stand - Find inspiration in the [community.md](./docs/community.md) file.

**Note:** Affiliate links are included; your support helps fund project development.

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
    *   `-W <waveshare device model>`: **Only** use this for Waveshare displays. Specify the model (e.g., `epd7in3f`).
    *   For Inky displays: `sudo bash install/install.sh`
    *   For Waveshare displays: `sudo bash install/install.sh -W epd7in3f`

After installation, reboot your Raspberry Pi. Refer to [installation.md](./docs/installation.md) for detailed instructions and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

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

To uninstall InkyPi, use:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More Plugins
*   Modular Layouts
*   Button Support
*   Improved Mobile UI

Explore upcoming features and vote on your favorites on the [Trello board](https://trello.com/b/SWJYWqe4/inkypi)!

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays.  **IT8951-based displays are not supported,** and screens smaller than 4 inches are not recommended.  Use the `-W` option during installation, specifying your display model.

## License

Distributed under the GPL 3.0 License ([LICENSE](./LICENSE)).

See [Attribution](./docs/attribution.md) for information on included fonts and icons.

## Troubleshooting & Support

*   Check the [troubleshooting guide](./docs/troubleshooting.md).
*   Report issues on [GitHub Issues](https://github.com/fatihak/InkyPi/issues).
*   Note: Known issues during Pi Zero W installation are detailed in the troubleshooting guide.

## Sponsoring

Support InkyPi's development:

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