# Lossless Scaling for Steam Deck: Enhance Your Gaming Experience

**Supercharge your Steam Deck gaming with this easy-to-use plugin that integrates Lossless Scaling's powerful frame generation technology.** [Check out the original repository here!](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk)

<p align="center">
   <img src="assets/decky-lossless-logo.png" alt="Lossless Scaling for Steam Deck Logo" width="200"/>
</p>

> **Note:** This is an **unofficial community plugin**. It is independently developed and **not officially supported** by the creators of Lossless Scaling or lsfg-vk. For support, please use the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3).

## Key Features:

*   **Effortless Installation:** Simplifies the setup of **lsfg-vk** (Lossless Scaling Frame Generation Vulkan layer) on your Steam Deck.
*   **Controller-Friendly UI:** Provides an intuitive interface for easy configuration within SteamOS or Bazzite.
*   **Customizable Settings:** Fine-tune your frame generation with options for FPS multiplier, flow scale, performance mode, and more.
*   **Real-time Configuration:** Apply changes instantly without restarting your game.
*   **HDR Support:** Enable HDR output for compatible games to enhance visuals.
*   **Experimental Features:** Experiment with advanced settings like present mode override and base FPS limit.
*   **Automatic lsfg-vk management**: Downloads and configures the Vulkan layer.
*   **Easy Uninstallation**: Cleanly removes all installed files.

## Installation Guide:

1.  **Purchase and install** [Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/) from Steam.
2.  **Download the plugin** from the [latest release](https://github.com/xXJSONDeruloXx/decky-lossless-scaling-vk/releases)
    - Download the "Lossless Scaling.zip" file to your Steam Deck
3.  **Install manually through Decky**:
    - In Game Mode, go to the settings cog in the top right of the Decky Loader tab
    - Enable "Developer Mode"
    - Go to "Developer" tab and select "Install Plugin from Zip"
    - Select the downloaded "Lossless Scaling.zip" file

## How to Use:

1.  Open the plugin from the Decky menu.
2.  Click "Install lsfg-vk" to automatically set up the Vulkan layer.
3.  Adjust your desired settings within the plugin's UI.
4.  Add `~/lsfg %command%` to your game's launch options in Steam Properties (or use the plugin's "Launch Option Clipboard" button).
5.  Launch your game to experience frame generation!

## Configuration Options:

### Core Settings:

*   **FPS Multiplier:** 2x, 3x, or 4x frame generation.
*   **Flow Scale:** Adjust motion estimation quality (lower = better performance, higher = better quality).
*   **Performance Mode:** Optimized for better performance.
*   **HDR Mode:** Enable for HDR-compatible games.

### Experimental Features:

*   **Present Mode Override:** Force specific Vulkan presentation modes.
*   **Base FPS Limit:** Set a base framerate cap.

## Troubleshooting:

**Frame Generation Not Working?**

*   Ensure you've added `~/lsfg %command%` to your game's launch options.
*   Check that the Lossless Scaling DLL was detected correctly in the plugin.
*   Try enabling Performance Mode.
*   Make sure your game is running in fullscreen mode.

**Performance Issues?**

*   Lower the Flow Scale setting.
*   Enable Performance Mode.
*   Reduce the FPS multiplier.
*   Consider using the experimental FPS limit.

## Credits:

*   **[PancakeTAS](https://github.com/PancakeTAS/lsfg-vk)** for creating the lsfg-vk Vulkan compatibility layer.
*   **[Lossless Scaling](https://store.steampowered.com/app/993090/Lossless_Scaling/)** developers.
*   **[Deck Wizard](https://www.youtube.com/@DeckWizard)** for the helpful video tutorial.
*   The **Decky Loader** team.
*   Community contributors and testers.

## Feedback and Support:

Join the [Decky Lossless Discord Channel](https://discord.gg/TwvHdVucC3) for community support and game-specific feedback.