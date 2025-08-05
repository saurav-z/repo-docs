# VRM Add-on for Blender: Create and Customize VRM Models within Blender

Easily import, export, and create VRM models directly within Blender with the VRM Add-on!  [View the source code on GitHub](https://github.com/saturday06/VRM-Addon-for-Blender).

## Key Features

*   **VRM Import & Export:** Seamlessly import and export VRM files for easy integration with other VRM-compatible platforms and applications.
*   **VRM Humanoid Support:** Add and configure VRM Humanoid rigs, enabling easy rigging and animation.
*   **MToon Shader Integration:** Easily configure the MToon shader for anime-style rendering within Blender.
*   **PBR Material Support:** Utilize Physics Based Rendering (PBR) materials for realistic VRM model visuals.
*   **Animation Tools:** Create and edit animations for your VRM models directly in Blender.
*   **Scripting API:** Automate tasks and extend functionality through Python scripting.

## Downloads

*   **Blender 4.2 or later:**  Get it from the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download from the [🌐**Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

Explore comprehensive tutorials to get you started:

|                                         [Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection)                                          |                                    [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection)                                    |                                    [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection)                                    |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
|                               **[Create Physics-Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection)**                               |                                     **[Create Anime-Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection)**                                     |                                      **[Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)**                                      |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |        <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a>        |
|                                        **[VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection)**                                         |                                           **[Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)**                                           |                                                                                                                                                                                        |
|    <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |         <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>         |                                                                                                                                                                                        |

## Development

The core addon files are located in the `src/io_scene_vrm` directory. For easier development, create a symbolic link to this directory within your Blender addons folder. Detailed instructions on setting up a development environment are available in the  [development how-to tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection).

**Setup symbolic link (example for Blender 4.2 or later):**

```bash
# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/extensions/user_default/vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/extensions/user_default/vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" src\io_scene_vrm
```

**(Adapt commands for older Blender versions or your OS as shown in the original README.)**

---