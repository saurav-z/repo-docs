# Enhance Your Steam Deck Gaming with Lossless Scaling Frame Generation

Supercharge your Steam Deck's performance with the **Decky Lossless Scaling** plugin, making frame generation effortless!  Visit the [original repository](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk) for the latest updates.

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

**Note:** This is an unofficial community plugin. It is independently developed and not officially supported by the creators of Lossless Scaling or lsfg-vk. For support, please use the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features

*   **Automated Installation:** Simplifies the installation of the **lsfg-vk** Vulkan layer.
*   **Controller-Friendly UI:**  Provides an intuitive interface within SteamOS for easy configuration.
*   **Customizable Settings:** Fine-tune frame generation with options for:
    *   FPS Multiplier (2x, 3x, 4x)
    *   Flow Scale
    *   Performance Mode
    *   HDR Mode
    *   Experimental Features (Present Mode Override, Base FPS Limit)
*   **Real-time Configuration:** Apply changes instantly without restarting your games.
*   **Easy Uninstallation:** Removes all installed files when no longer needed.

## Installation Guide

1.  **Purchase and Install Lossless Scaling:** Requires [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) from Steam.
2.  **Download the Plugin:** Download the "Lossless Scaling.zip" file from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases) to your Steam Deck.
3.  **Install via Decky Loader:**
    *   Enter Game Mode and access the Decky Loader settings.
    *   Enable "Developer Mode."
    *   Navigate to the "Developer" tab and choose "Install Plugin from Zip."
    *   Select the downloaded "Lossless Scaling.zip" file.
4.  **Install lsfg-vk:** Open the plugin from the Decky menu and click "Install lsfg-vk"

## How to Use

1.  **Configure the plugin:** Adjust settings in the plugin's UI (FPS multiplier, flow scale, etc.).
2.  **Apply Launch Option:** Add `~/lsfg %command%` to your game's launch options or use the "Launch Option Clipboard" button within the plugin.
3.  **Launch Your Game:** Frame generation will activate automatically based on your settings.

## Troubleshooting

**Frame generation not working?**

*   Ensure you've added `~/lsfg %command%` to your game's launch options.
*   Confirm the Lossless Scaling DLL was correctly detected in the plugin.
*   Try enabling Performance Mode.
*   Ensure your game is running in fullscreen mode.

**Performance Issues?**

*   Lower the Flow Scale.
*   Enable Performance Mode.
*   Reduce the FPS multiplier.
*   Consider using the experimental FPS limit feature.

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** - Creator of the lsfg-vk Vulkan layer
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) developers** - For the original frame generation technology
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** - For helpful video tutorial
*   **Decky Loader Team** - For the plugin framework
*   **Community Contributors & Testers** - For feedback and bug reports