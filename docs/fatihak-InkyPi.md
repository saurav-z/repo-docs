# InkyPi: Your Customizable E-Ink Dashboard for Raspberry Pi

**Transform your Raspberry Pi into a sleek, low-power display with InkyPi, showcasing the information you need in a beautiful, paper-like format. [Visit the original repository](https://github.com/fatihak/InkyPi) for more details.**

<img src="./docs/images/inky_clock.jpg" alt="InkyPi E-Ink Display"/>

## Key Features

*   **E-Ink Elegance:** Experience crisp, easy-on-the-eyes visuals with no glare or backlight, mimicking the look of real paper.
*   **Web-Based Control:** Effortlessly configure and update your display from any device on your network using an intuitive web interface.
*   **Distraction-Free Design:** Enjoy a minimalist experience with no LEDs, noise, or distracting notifications.
*   **Easy Setup:** Get started quickly with straightforward installation and configuration, perfect for beginners and experienced makers.
*   **Open Source & Customizable:** Modify, extend, and create your own plugins to tailor InkyPi to your specific needs.
*   **Scheduled Playlists:** Display different content at designated times with playlist functionality.

## Available Plugins

*   **Image Upload:** Display your favorite images.
*   **Daily Newspaper/Comic:** Get your daily dose of news and funnies.
*   **Clock:** Customize your clock faces for the perfect time display.
*   **AI Image/Text:** Generate images and text using OpenAI's models.
*   **Weather:** Stay informed with current conditions and forecasts.
*   **Calendar:** Visualize your calendar events from various providers.

*...and more plugins are constantly being added!*  See [Building InkyPi Plugins](./docs/building_plugins.md) for information on creating your own.

## Hardware Requirements

*   **Raspberry Pi:** (4 | 3 | Zero 2 W)
    *   Recommended: 40-pin Pre-Soldered Header
*   **MicroSD Card:** (min 8 GB) - *Example: [Amazon Link](https://amzn.to/3G3Tq9W)*
*   **E-Ink Display:**

    *   **Pimoroni Inky Impression:** (Various Sizes)
        *   [13.3 Inch Display](https://collabs.shop/q2jmza)
        *   [7.3 Inch Display](https://collabs.shop/q2jmza)
        *   [5.7 Inch Display](https://collabs.shop/ns6m6m)
        *   [4 Inch Display](https://collabs.shop/cpwtbh)
    *   **Pimoroni Inky wHAT:** (4.2 Inch)
        *   [4.2 Inch Display](https://collabs.shop/jrzqmf)
    *   **Waveshare e-Paper Displays:** (Various Sizes & Colors)
        *   Spectra 6 (E6) Full Color: [4 inch](https://www.waveshare.com/4inch-e-paper-hat-plus-e.htm?&aff_id=111126), [7.3 inch](https://www.waveshare.com/7.3inch-e-paper-hat-e.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-plus-e.htm?&aff_id=111126)
        *   Black and White: [7.5 inch](https://www.waveshare.com/7.5inch-e-paper-hat.htm?&aff_id=111126), [13.3 inch](https://www.waveshare.com/13.3inch-e-paper-hat-k.htm?&aff_id=111126)
        *   See [Waveshare e-paper displays](https://www.waveshare.com/product/raspberry-pi/displays/e-paper.htm?&aff_id=111126) or visit their [Amazon store](https://amzn.to/3HPRTEZ) for additional models.
*   **Picture Frame or 3D Stand:**  Find inspiration and designs in [community.md](./docs/community.md).

    *Affiliate Disclosure: *  The links above are affiliate links that support the project at no extra cost to you.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
3.  **Run the installation script (with `sudo`):**
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`:  Specify this **ONLY** for Waveshare displays. Provide the model e.g., `epd7in3f`.

    *   **Inky Displays:**
        ```bash
        sudo bash install/install.sh
        ```
    *   **Waveshare Displays:**
        ```bash
        sudo bash install/install.sh -W epd7in3f
        ```

After installation, reboot your Raspberry Pi. The display will show the InkyPi splash screen.  For more details, consult [installation.md](./docs/installation.md) and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating

1.  **Navigate to the project directory:**
    ```bash
    cd InkyPi
    ```
2.  **Fetch latest changes:**
    ```bash
    git pull
    ```
3.  **Run the update script (with `sudo`):**
    ```bash
    sudo bash install/update.sh
    ```

## Uninstalling

```bash
sudo bash install/uninstall.sh
```

## Roadmap

*   More Plugins!
*   Modular Layouts
*   Button Support
*   Improved Web UI for Mobile

Explore the [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to contribute and get involved!

## Waveshare Display Compatibility

InkyPi supports Waveshare e-Paper displays.  **IT8951 controller-based displays are *not* supported**, and displays smaller than 4 inches are not recommended.  Use the `-W` option during installation with your display model.

## License

Distributed under the GPL 3.0 License.  See [LICENSE](./LICENSE).  See [Attribution](./docs/attribution.md) for details on fonts and icons.

## Troubleshooting & Support

Check the [troubleshooting guide](./docs/troubleshooting.md) for solutions. If you have issues, please open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

## Sponsoring

Support the InkyPi project!
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