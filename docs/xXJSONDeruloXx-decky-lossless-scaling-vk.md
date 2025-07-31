# Enhance Your Steam Deck Gaming with Lossless Scaling Frame Generation (LSFG)

Unlock smoother gameplay on your Steam Deck with the **Lossless Scaling** plugin, a streamlined tool to easily install and configure frame generation using [lsfg-vk](https://github.com/PancakeTAS/lsfg-vk), bringing you a controller-friendly experience in SteamOS or Bazzite. 

[View the original repo on GitHub](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk)

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

> **Important Note:** This is an unofficial community plugin and is **not officially supported** by the creators of Lossless Scaling or lsfg-vk. For support, please visit the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features:

*   **Simplified Installation:** Automates the setup of the lsfg-vk Vulkan layer on your Steam Deck.
*   **User-Friendly Interface:** Easily configure frame generation settings through the Decky plugin UI.
*   **Real-time Configuration:** Adjust settings on the fly, without restarting your game.
*   **Configurable FPS Multiplier:** Choose between 2x, 3x, or 4x frame generation.
*   **Motion Estimation Control:** Fine-tune "Flow Scale" for optimal balance between performance and quality.
*   **Performance Mode:** Optimize processing for improved performance in most games.
*   **HDR Support:** Enable HDR mode for compatible games.
*   **Experimental Features:** Access advanced options like Present Mode Override and Base FPS Limit.
*   **Automatic DLL Detection:** Automatically detects your Lossless Scaling DLL installation.
*   **Easy Uninstallation:** Cleanly removes all installed files.

## Installation Guide:

1.  **Download the Plugin:** Get the latest release from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases) and download the "Lossless Scaling.zip" file to your Steam Deck.
2.  **Install via Decky Loader:**
    *   In Game Mode, go to the settings cog in the top right of the Decky Loader tab.
    *   Enable "Developer Mode".
    *   Go to the "Developer" tab and select "Install Plugin from Zip".
    *   Select the downloaded "Lossless Scaling.zip" file.

## How to Use:

1.  **Purchase and Install Lossless Scaling:** Get the [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) application from Steam.
2.  **Open the Plugin:** Access the plugin from the Decky menu.
3.  **Install lsfg-vk:** Click "Install lsfg-vk" within the plugin to automatically set up the Vulkan layer.
4.  **Configure Settings:** Customize your frame generation settings in the plugin UI (FPS multiplier, flow scale, performance mode, HDR settings, etc.).
5.  **Apply Launch Option:** Add `~/lsfg %command%` to your game's launch options in Steam Properties (or use the "Launch Option Clipboard" button).
6.  **Launch Your Game:** Frame generation will activate automatically using your plugin configuration.

## Configuration Options Explained:

### Core Settings:

*   **FPS Multiplier:**  Select 2x, 3x, or 4x frame generation.
*   **Flow Scale:** Adjust motion estimation quality; lower values prioritize performance.
*   **Performance Mode:** Recommended for most games to optimize processing.
*   **HDR Mode:** Enable for games that support HDR output.

### Experimental Features:

*   **Present Mode Override:** Force specific Vulkan presentation modes.
*   **Base FPS Limit:** Cap your base framerate before applying the multiplier (useful for DirectX games).

## Troubleshooting:

**Frame Generation Not Working?**

*   Verify you've added `LSFG_PROCESS=decky-lsfg-vk %command%` to your game's launch options.
*   Confirm Lossless Scaling DLL detection in the plugin.
*   Try Performance Mode if you're experiencing crashes.
*   Ensure your game is running in fullscreen.

**Performance Issues?**

*   Reduce the Flow Scale setting.
*   Enable Performance Mode.
*   Lower the FPS multiplier.
*   Use the experimental FPS limit for DirectX games.

## Credits

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for creating the lsfg-vk Vulkan compatibility layer
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers for the original frame generation technology
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** for the helpful video tutorial
*   The **Decky Loader** team for the plugin framework
*   Community contributors and testers for feedback and bug reports