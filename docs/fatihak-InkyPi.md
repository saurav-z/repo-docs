# InkyPi: Your Customizable E-Ink Display for Raspberry Pi

[![InkyPi Display](docs/images/inky_clock.jpg)](https://github.com/fatihak/InkyPi)

**Transform your Raspberry Pi into a low-power, distraction-free display with InkyPi, the open-source e-ink solution.**

## Key Features

*   **Paper-Like Aesthetics:** Enjoy crisp, glare-free visuals with e-ink technology.
*   **Web Interface:** Easily configure and update your display from any device on your network.
*   **Minimal Distractions:** No LEDs, noise, or notifications – just the content you need.
*   **Easy Setup:** Straightforward installation and configuration, perfect for beginners.
*   **Open Source:** Modify, customize, and create your own plugins.
*   **Scheduled Playlists:** Display different content at specific times.

## Plugins

InkyPi offers a variety of plugins to display the information you care about.  More plugins are constantly being added!

*   **Image Upload:** Display your favorite images.
*   **Daily Newspaper/Comic:** Stay up-to-date with news and comics.
*   **Clock:** Customize your clock faces.
*   **AI Image/Text:** Generate images and text using AI.
*   **Weather:** Monitor current conditions and forecasts.
*   **Calendar:** Visualize your calendar events.

For information on creating custom plugins, see the [Building InkyPi Plugins](./docs/building_plugins.md) documentation.

## Hardware

*   **Raspberry Pi** (4, 3, or Zero 2 W) - Pre-soldered header recommended.
*   **MicroSD Card** (min 8 GB) - [Example Link](https://amzn.to/3G3Tq9W)
*   **E-Ink Display:**
    *   **Pimoroni Inky Impression**
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT**
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays**
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   Browse more models on [Waveshare's e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or their [Amazon store](https://amzn.to/3HPRTEZ).
        *   **Note:** IT8951-based displays are not supported.

*   **Picture Frame or 3D Stand**
    *   Check out [community.md](./docs/community.md) for community contributions.

**Affiliate Disclosure:**  Some links above are affiliate links.  Commissions earned help support this project at no extra cost to you.

## Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the Installation Script (with `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`: **Required ONLY for Waveshare displays.** Specify the Waveshare model (e.g., `epd7in3f`).

    *   **Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **Waveshare Displays:**
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

    After installation, reboot your Raspberry Pi. The display will show the InkyPi splash screen.

    **Important:** The script requires `sudo`.  Start with a fresh Raspberry Pi OS installation to avoid conflicts. See [installation.md](./docs/installation.md) for more details and a [YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Update

1.  **Navigate to the Project Directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch Latest Changes:**
    ```bash
    git pull
    ```
3.  **Run the Update Script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

This updates your InkyPi installation to the latest version.

## Uninstall

Remove InkyPi using the following command:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More plugins!
*   Modular layout customization.
*   Button support for interactive displays.
*   Improved mobile Web UI.

See the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) for the latest feature planning.

## Waveshare Display Support

InkyPi supports various Waveshare e-Paper displays. **IT8951-based displays are NOT supported.** **Screens smaller than 4 inches are NOT recommended.**

Specify your display model during installation using the `-W` option. The script will automatically install the necessary driver if it's available in the [Waveshare Python EPD library](https://github.com/waveshareteam/e-Paper/tree/master/RaspberryPi_JetsonNano/python/lib/waveshare_epd).

## License

Distributed under the GPL 3.0 License. See [LICENSE](./LICENSE).

Font and icon licensing information is in [Attribution](./docs/attribution.md).

## Troubleshooting

Check the [troubleshooting guide](./docs/troubleshooting.md). If you're still facing issues, create an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W users:** See the [Known Issues during Pi Zero W Installation](./docs/troubleshooting.md#known-issues-during-pi-zero-w-installation) section for details.

## Sponsoring

Support InkyPi's ongoing development:

<p align="center">
<a href="https://github.com/sponsors/fatihak" target="_blank"><img src="https://user-images.githubusercontent.com/345274/133218454-014a4101-b36a-48c6-a1f6-342881974938.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.patreon.com/akzdev" target="_blank"><img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patreon" height="35" width="auto"></a>
<a href="https://www.buymeacoffee.com/akzdev" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="35" width="auto"></a>
</p>

## Acknowledgements

*   [PaperPi](https://github.com/txoof/PaperPi) - Supports Waveshare devices.
*   [InkyCal](https://github.com/aceinnolab/Inkycal) - Modular plugins.
*   [PiInk](https://github.com/tlstommy/PiInk) - Inspiration for the web UI.
*   [rpi_weather_display](https://github.com/sjnims/rpi_weather_display) - Alternative e-ink weather dashboard.

[Back to Top](#inkypi-your-customizable-e-ink-display-for-raspberry-pi)  |  [GitHub Repo](https://github.com/fatihak/InkyPi)
```
Key improvements:

*   **SEO Optimization:**  Includes keywords like "Raspberry Pi," "e-ink," and mentions of key functionalities within the headings and content.
*   **Clear Headings and Structure:** Uses clear, concise headings for readability and organization.
*   **Bulleted Key Features:**  Highlights the main benefits in an easily digestible format.
*   **Concise Language:**  Streamlines the descriptions for better understanding.
*   **Actionable Installation Steps:** The "Installation" section is improved to guide the user.
*   **Call to Action (Sponsoring):**  Keeps the sponsor information front-and-center.
*   **Back to Top and Repo links**: Make it easy to navigate