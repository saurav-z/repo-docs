# InkyPi: Your Customizable E-Ink Display

[<img src="./docs/images/inky_clock.jpg" alt="InkyPi Clock" width="500">](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a low-power, distraction-free display with InkyPi, a customizable E-Ink solution that keeps you informed.**

## Key Features

*   **Eye-Friendly Display:** Enjoy crisp, paper-like visuals with no glare or backlighting.
*   **Web-Based Control:** Effortlessly configure and update your display from any device on your network.
*   **Minimalist Design:** Stay focused with a display free of LEDs, noise, and distracting notifications.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration for beginners and makers.
*   **Open Source & Extensible:** Modify, customize, and create your own plugins to suit your needs.
*   **Scheduled Playlists:** Display a rotating selection of plugins at designated times.

## Plugins

InkyPi offers a range of built-in plugins, with more on the way!

*   **Image Upload:** Display any image you choose.
*   **Daily Newspaper/Comic:** View daily comics and front pages from major newspapers.
*   **Clock:** Choose from a selection of customizable clock faces.
*   **AI Image/Text:** Generate images and text from OpenAI's models.
*   **Weather:** See current and forecast weather conditions.
*   **Calendar:** Visualize your calendar from Google, Outlook, or Apple Calendar.

For information on building custom plugins, see [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware

You'll need the following hardware to get started:

*   **Raspberry Pi:** (4 | 3 | Zero 2 W) - A pre-soldered header is recommended.
*   **MicroSD Card:** (min 8 GB), like [this one](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:** Choose from a variety of supported displays:

    *   **Pimoroni Inky Impression:**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:**
        *   Spectra 6 (E6) Full Color [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126) / [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126) / [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126) / [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models. Note that some models like the IT8951 based displays are not supported. See later section on [Waveshare e-Paper](#waveshare-display-support) compatibilty for more information.

*   **Optional:** Picture Frame or 3D Stand. Find community-created designs and inspiration in [community.md](./docs/community.md).

**Affiliate Disclosure:** Some links above are affiliate links. Commissions earned support the project at no extra cost to you.

## Installation

Get InkyPi up and running with these simple steps:

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
    *   **For Waveshare Displays:**  Specify your model during installation.
        ```bash
        sudo bash install/install.sh -W <waveshare device model>
        ```
        Replace `<waveshare device model>` (e.g., `epd7in3f`) with your display's model.

After installation, reboot your Raspberry Pi. The display will update to show the InkyPi splash screen.

**Important Notes:**

*   The installation script requires `sudo` privileges.
*   We recommend starting with a fresh Raspberry Pi OS installation.
*   The script automatically enables necessary interfaces (SPI and I2C).
*   Refer to [installation.md](./docs/installation.md) for detailed instructions and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

Keep your InkyPi up-to-date:

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script:**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstall

Remove InkyPi with a single command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is continually evolving! Explore upcoming features and vote on your favorites:

*   More Plugins
*   Modular layouts for mixing and matching plugins
*   Button Support
*   Improved Mobile UI

Check out the [Trello Board](https://trello.com/b/SWJYWqe4/inkypi) to stay informed.

## Waveshare Display Support

InkyPi supports a variety of Waveshare e-Paper displays.  **Displays based on the IT8951 controller are not supported**. **Screens smaller than 4 inches are not recommended** due to limited resolution.

*   Ensure your Waveshare display model has a corresponding driver in their [Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).
*   Use the `-W` option with the correct model during installation (e.g., `-W epd7in3f`).

## License

Distributed under the [GPL 3.0 License](./LICENSE).

See [Attribution](./docs/attribution.md) for font and icon licensing details.

## Troubleshooting

Find solutions in the [troubleshooting guide](./docs/troubleshooting.md).  Report issues on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Known Issue:** Users of Pi Zero W may encounter issues during installation. See the troubleshooting guide for details.

## Sponsoring

Support InkyPi's development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

Inspired by and/or builds upon the work of:

*   [PaperPi](https://github.com/txoof/PaperPi) - Supports Waveshare devices.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugin dashboards.
*   [PiInk](https://github.com/tlstommy/PiInk) - Flask web UI inspiration.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - eink weather dashboard with power efficiency.

---

**Get started with InkyPi today and experience the power of a custom E-Ink display!**