# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="600"/>

**Transform your Raspberry Pi into a low-power, distraction-free display with InkyPi, an open-source platform for showcasing the information that matters most to you.** [View the original repository on GitHub](https://github.com/fatihak/InkyPi).

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting, perfect for any environment.
*   **Web-Based Control:** Effortlessly manage your display from any device on your network using a simple web interface.
*   **Minimalist & Focused:** Eliminate distractions with a display that's free of LEDs, noise, and unwanted notifications.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration, ideal for both beginners and makers.
*   **Open Source & Extensible:** Customize and expand InkyPi with your own plugins and modifications.
*   **Scheduled Playlists:** Set up playlists to display different content at specific times.

## Plugins Available

*   Image Upload
*   Daily Newspaper/Comic
*   Clock
*   AI Image/Text Generation
*   Weather
*   Calendar

*   **More plugins coming soon!** For information on custom plugin development, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) -  40-pin Pre-Soldered Header recommended.
*   **MicroSD Card:** Minimum 8 GB, such as [this one](https://amzn.to/3G3Tq9W).
*   **E-Ink Display:** Compatible with:
    *   Inky Impression by Pimoroni (13.3", 7.3", 5.7", and 4" displays - [links](https://collabs.shop/q2jmza))
    *   Inky wHAT by Pimoroni (4.2" display - [link](https://collabs.shop/jrzqmf))
    *   Waveshare e-Paper Displays (Spectra 6 Full Color and Black and White models -  [links](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126))
    *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ). **Note:** IT8951 based displays are not supported.  See [Waveshare e-Paper](#waveshare-display-support) for more information.
*   **Picture Frame or 3D Stand:** Find inspiration and community submissions in [community.md](./docs/community.md).

**Disclosure:** This project uses affiliate links.

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
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```

    *   **-W `<waveshare device model>`:**  Use this option ONLY when installing for a Waveshare display.  Specify the Waveshare model (e.g., `epd7in3f`).

    *   **For Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **For Waveshare Displays:**
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

    After installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

    **Important Notes:**

    *   Requires sudo privileges.
    *   Recommended to start with a fresh Raspberry Pi OS installation.
    *   Automatically enables SPI and I2C interfaces.

    For detailed instructions, including microSD card imaging, see [installation.md](./docs/installation.md) and the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

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
    This ensures all code changes and dependencies are correctly applied.

## Uninstalling InkyPi

```bash
sudo bash install/uninstall.sh
```

## Roadmap & Future Development

InkyPi is actively being improved, with plans for:

*   More plugins
*   Modular layouts for customized displays
*   Button support with customizable actions
*   Improved Web UI for mobile devices

Track upcoming features and vote on your favorites on the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

This project supports Waveshare e-Paper displays. **IT8951 controller-based displays are not supported**. Screens smaller than 4 inches are not recommended due to resolution limitations.

If your display model has a driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd), it should be compatible.

When running the installation script, use the `-W` option followed by your display model (without the `.py` extension).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for details.

This project uses fonts and icons with separate licenses. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting & Support

Check the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues. If you encounter problems, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Note:** Known issues exist with Pi Zero W installations. See the troubleshooting guide for details.

## Sponsoring

Support InkyPi's continued development!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and similar to:

*   [PaperPi](https://github.com/txoof/PaperPi)
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)