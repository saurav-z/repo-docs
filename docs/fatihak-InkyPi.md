# InkyPi: Your Customizable E-Ink Display for a Distraction-Free Life

<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock Display" />

Transform your Raspberry Pi into a beautiful and informative display with InkyPi, an open-source project that brings the elegance of E-Ink to your desktop. [View the original repository](https://github.com/fatihak/InkyPi).

## Key Features:

*   **Paper-like Aesthetic:** Enjoy crisp, minimalist visuals with no glare or backlighting for a comfortable viewing experience.
*   **Web-Based Control:** Configure and update your display from any device on your network via a user-friendly web interface.
*   **Minimalist Design:** Eliminate distractions with an E-Ink display that's free of LEDs, noise, and unwanted notifications.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration, perfect for beginners and makers.
*   **Open Source:** Customize and extend InkyPi by creating your own plugins or modifying existing ones.
*   **Scheduled Playlists:** Automatically display different content at specific times with customizable playlists.

## Core Plugins:

*   **Image Upload:** Display any image from your web browser.
*   **Daily Newspaper/Comic:** Stay updated with daily comics and front pages of major newspapers.
*   **Clock:** Choose from a variety of customizable clock faces.
*   **AI Image/Text:** Generate images and dynamic text using OpenAI's models.
*   **Weather:** View current weather conditions and forecasts with a custom layout.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

And more plugins are on the way! For information on building custom plugins, see the [Building InkyPi Plugins documentation](./docs/building_plugins.md).

## Hardware Requirements:

*   Raspberry Pi (4 | 3 | Zero 2 W) - *Recommended to get 40 pin Pre Soldered Header*
*   MicroSD Card (min 8 GB) - [Example](https://amzn.to/3G3Tq9W)
*   E-Ink Display:

    *   Inky Impression by Pimoroni
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   Inky wHAT by Pimoroni
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   Waveshare e-Paper Displays
        *   Spectra 6 (E6) Full Color
            *   [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126)
            *   [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White
            *   [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126)
            *   [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.
*   Picture Frame or 3D Stand
    *   See [community.md](./docs/community.md) for 3D models, custom builds, and other submissions from the community

**Disclosure:** The above links are affiliate links. I may earn a commission from qualifying purchases made through them, at no extra cost to you, which helps maintain and develop this project.

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
    *   **-W \<waveshare device model\>** - Specify this parameter **ONLY** if installing for a Waveshare display. After the -W option specify the Waveshare device model e.g. epd7in3f.
    *   For Inky displays:

        ```bash
        sudo bash install/install.sh
        ```
    *   For Waveshare displays:

        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, the script prompts for a reboot. Your display will then show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires sudo privileges.
*   Start with a fresh Raspberry Pi OS installation to avoid potential conflicts.
*   The script enables the necessary SPI and I2C interfaces.

For detailed installation instructions, including microSD card imaging, consult [installation.md](./docs/installation.md) or watch [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

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

## Uninstalling InkyPi

Run the uninstall script:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is continuously evolving! Future plans include:

*   More plugins!
*   Modular layouts for mixing and matching plugins
*   Button support with customizable action bindings
*   Improved Web UI on mobile devices

Explore and vote on upcoming features on the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi)!

## Waveshare Display Support

InkyPi also supports Waveshare e-Paper displays. While similar to Pimoroni's Inky screens, Waveshare displays use specific drivers from their [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).

**Compatibility:**

*   Tested with various Waveshare models.
*   **IT8951 controller-based displays are NOT supported.**
*   **Screens smaller than 4 inches are not recommended** due to limited resolution.

If your display has a driver in the linked library, it's likely compatible. Use the `-W` option with your display model during installation.

## License

Distributed under the GPL 3.0 License, see [LICENSE](./LICENSE).

Includes fonts and icons with separate licensing and attribution requirements. See [Attribution](./docs/attribution.md) for details.

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md). If issues persist, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W Issues:** See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section in the troubleshooting guide.

## Sponsoring

Support InkyPi's development!

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Related projects:

*   [PaperPi](https://github.com/txoof/PaperPi)
    *   Shoutout to @txoof for assisting with InkyPi's installation process
*   [InkyCal](https://github.com/aceinnolab/Inkycal)
*   [PiInk](https://github.com/tlstommy/PiInk)
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display)