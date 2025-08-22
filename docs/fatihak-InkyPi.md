# InkyPi: Your Customizable E-Ink Display for a Paper-Like Experience

[View the original repository on GitHub](https://github.com/fatihak/InkyPi)

InkyPi is an open-source project that transforms a Raspberry Pi and an e-ink display into a sleek, energy-efficient information hub, perfect for displaying the content you care about with a minimalist aesthetic.

## Key Features

*   **Paper-Like Display:** Enjoy crisp, clear visuals with no glare or backlight, providing a comfortable viewing experience.
*   **Web-Based Interface:** Easily configure and update your display from any device on your network using a user-friendly web interface.
*   **Minimalist Design:** Minimize distractions with a display free of LEDs, noise, and notifications, keeping your focus on the essential information.
*   **Easy Setup:** Quickly install and configure InkyPi, perfect for beginners and experienced makers alike.
*   **Open Source:** Customize, modify, and create your own plugins to tailor your display to your specific needs.
*   **Scheduled Playlists:** Arrange different plugins to run at designated times, showcasing a dynamic flow of information.

## Plugins

InkyPi offers a range of plugins to display various types of information:

*   **Image Upload:** Display any image you want.
*   **Daily Newspaper/Comic:** View your daily news and comics.
*   **Clock:** Customize your clock face.
*   **AI Image/Text:** Generate images and text from prompts using OpenAI models.
*   **Weather:** See current conditions and forecasts.
*   **Calendar:** Sync your Google, Outlook, or Apple calendar.

New plugins are consistently being added! You can find documentation on creating custom plugins at [Building InkyPi Plugins](./docs/building_plugins.md).

## Hardware Requirements

*   **Raspberry Pi:** (4, 3, or Zero 2 W)
*   **MicroSD Card:** (min 8 GB) ([Recommended Link](https://amzn.to/3G3Tq9W))
*   **E-Ink Display:**

    *   **Pimoroni Inky Impression:** (13.3", 7.3", 5.7", 4")
    *   **Pimoroni Inky wHAT:** (4.2")
    *   **Waveshare e-Paper Displays:** (Various sizes - see below)
*   **Optional:** Picture Frame or 3D Stand ([Community Resources](./docs/community.md))

**Note:** Affiliate links are included; purchases through these links may help support the project.

## Installation

To install InkyPi:

1.  Clone the repository:
    ```bash
    git clone https://github.com/fatihak/InkyPi.git
    ```
2.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
3.  Run the installation script with `sudo`:
    ```bash
    sudo bash install/install.sh [-W <waveshare device model>]
    ```
    *   `-W <waveshare device model>`: Use this **ONLY** for Waveshare displays; specify the model (e.g., `epd7in3f`).

    Examples:
    *   For Inky displays: `sudo bash install/install.sh`
    *   For Waveshare displays: `sudo bash install/install.sh -W epd7in3f`

After installation, reboot your Raspberry Pi. The display will then show the InkyPi splash screen.

**Important:**

*   The installation script requires `sudo` privileges. Start with a fresh Raspberry Pi OS installation to avoid potential conflicts.
*   The script will automatically enable necessary interfaces.

See [installation.md](./docs/installation.md) for details, including how to image your microSD card, and [this YouTube tutorial](https://youtu.be/L5PvQj1vfC4).

## Updating InkyPi

1.  Navigate to the project directory:
    ```bash
    cd InkyPi
    ```
2.  Fetch latest changes:
    ```bash
    git pull
    ```
3.  Run update script:
    ```bash
    sudo bash install/update.sh
    ```

This ensures your InkyPi stays up-to-date without reinstallation.

## Uninstalling

To uninstall InkyPi:

```bash
sudo bash install/uninstall.sh
```

## Roadmap

InkyPi is constantly evolving, with future plans that include:

*   More plugins
*   Modular layouts
*   Button support
*   Improved Web UI

Check out the public [Trello board](https://trello.com/b/SWJYWqe4/inkypi) to see upcoming features and vote on what you want!

## Waveshare Display Support

InkyPi supports Waveshare e-Paper displays.  **IT8951 controller-based displays are NOT supported.**

When installing with a Waveshare display, use the `-W` option with the specific model (e.g., `epd7in3f`).  The script will install the correct drivers.  Screens smaller than 4" are not recommended.

## License

InkyPi is licensed under the GPL 3.0 License ([LICENSE](./LICENSE)). Fonts and icons have separate licensing (see [Attribution](./docs/attribution.md)).

## Troubleshooting

Check out the [troubleshooting guide](./docs/troubleshooting.md). If you still need help, open an issue on the [GitHub Issues](https://github.com/fatihak/InkyPi/issues) page.

**Pi Zero W Users:** See the troubleshooting guide for known installation issues.

## Sponsoring

Support the continued development of InkyPi!

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