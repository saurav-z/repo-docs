# InkyPi: Your Customizable E-Ink Display for a Paper-Like Experience

[<img src="./docs/images/inky_clock.jpg" alt="InkyPi Display" width="500"/>](https://github.com/fatihak/InkyPi)

Transform your Raspberry Pi into a sleek, low-power information hub with InkyPi, an open-source project that brings dynamic content to life on beautiful E-Ink displays.

**Key Features:**

*   **Eye-Friendly Display:** Experience crisp, minimalist visuals with a natural paper-like aesthetic, free from glare and backlighting.
*   **Web-Based Configuration:** Easily set up and customize your display from any device on your network using a simple web interface.
*   **Minimal Distractions:** Enjoy a distraction-free experience with no LEDs, noise, or notifications, just the content you need.
*   **Beginner-Friendly Setup:** Get up and running quickly with straightforward installation and configuration.
*   **Open-Source Customization:** Modify, personalize, and create your own plugins to tailor the display to your exact needs.
*   **Scheduled Playlists:** Set up playlists to display different plugins at designated times, managing your content effortlessly.

**Plugins:**

*   Image Upload: Display any image from your browser.
*   Daily Newspaper/Comic: Show daily comics and front pages from around the world.
*   Clock: Customize clock faces for displaying the time.
*   AI Image/Text: Generate images and dynamic text using OpenAI's models.
*   Weather: Display current weather conditions and multi-day forecasts.
*   Calendar: Visualize your calendar from Google, Outlook, or Apple Calendar.

And many more plugins are coming soon! For documentation on building custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   Raspberry Pi (4 | 3 | Zero 2 W)
    *   Recommended to get a 40-pin Pre-Soldered Header
*   MicroSD Card (min 8 GB) - e.g., [Amazon Link](https://amzn.to/3G3Tq9W)
*   E-Ink Display:
    *   Inky Impression by Pimoroni
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   Inky wHAT by Pimoroni
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   Waveshare e-Paper Displays
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   Picture Frame or 3D Stand
    *   See [community.md](./docs/community.md) for community submissions.

**Note:** These are affiliate links, and using them helps support the project.

## Installation

Follow these steps to install InkyPi:

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
    *   `-W <waveshare device model>` - Specify this parameter **ONLY** for Waveshare displays.  Specify the Waveshare device model e.g. `epd7in3f`.

    *   For Inky displays:
        ```bash
        sudo bash install/install.sh
        ```
    *   For Waveshare displays:
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```
After installation, reboot your Raspberry Pi. The display will update to show the InkyPi splash screen.

**Important:**

*   The installation script requires sudo privileges. Start with a fresh Raspberry Pi OS installation to avoid conflicts.
*   The script enables necessary SPI and I2C interfaces.

For more details, refer to [installation.md](./docs/installation.md) and the [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

To update to the latest code changes:

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

## Uninstalling InkyPi

To uninstall:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

The InkyPi project is continuously evolving.  Planned features include:

*   More Plugins!
*   Modular layouts.
*   Button support with customizable actions.
*   Improved Web UI on mobile devices.

Explore upcoming features and vote on your favorites on the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi).

## Waveshare Display Support

InkyPi now supports Waveshare e-Paper displays.  **IT8951-based displays are not supported**, and **displays smaller than 4 inches are not recommended.**

To use a Waveshare display:

1.  Ensure your model has a driver in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).
2.  Use the `-W` option during installation, specifying your display model (e.g., `-W epd7in3f`).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE) for more information.

This project includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md) for solutions to common issues.  If you still need help, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

If you're using a Pi Zero W, see the [troubleshooting guide](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) for known installation issues.

## Sponsoring

Support the development of InkyPi!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a GitHub Sponsor" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Special thanks to:

*   [PaperPi](https://github.com/txoof/PaperPi) - and @txoof for assistance with the installation process.
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)