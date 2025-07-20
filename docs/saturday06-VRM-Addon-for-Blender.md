# VRM Add-on for Blender: Create and Customize VRM Models with Ease

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

This Blender add-on empowers you to import, export, and customize VRM models directly within Blender, streamlining your 3D character workflow.  **[View the project on GitHub](https://github.com/saturday06/VRM-Addon-for-Blender)**

## Key Features

*   **Import and Export VRM:** Seamlessly bring VRM models into Blender and export your creations in the VRM format.
*   **VRM Humanoid Support:** Easily add and configure VRM Humanoid rigs for your characters.
*   **MToon Shader Integration:** Utilize the popular MToon shader for creating anime-style materials.
*   **Material Customization:** Create Physics Based Rendering (PBR) and anime-style materials.
*   **Animation Support:** Animate your VRM models within Blender.
*   **Scripting API:** Automate tasks and extend functionality with Python scripts.

## Download

*   **Blender 4.2 or later:** [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm)
*   **Blender 2.93 to 4.1:** [🌐**The Official Site**](https://vrm-addon-for-blender.info)

## Tutorials

Explore the following tutorials to get started:

|                                         Installation                                           |                                  Create Simple VRM                                    |                                   Create Humanoid VRM                                    |
| :---------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------: |
|   [Installation](https://vrm-addon-for-blender.info/en/installation?locale_redirection)   | [Create Simple VRM](https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection) | [Create Humanoid VRM](https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection) |
|                                  Create Physics Based Material                                 |                              Create Anime Style Material                               |                                     Automation with Python Scripts                                     |
| [Create Physics Based Material](https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection) |   [Create Anime Style Material](https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection)   |   [Automation with Python Scripts](https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection)   |
|                                          VRM Animation                                          |                                        Development How-To                                        |                                                                                                |
|    [VRM Animation](https://vrm-addon-for-blender.info/en/animation?locale_redirection)    |        [Development How-To](https://vrm-addon-for-blender.info/en/development?locale_redirection)        |                                                                                                |

## Development

The core source code resides in the `main` branch, specifically within the `src/io_scene_vrm` folder.  For efficient development, consider linking this folder within Blender's addon directory. Learn more about advanced development using [astral.sh/uv](https://docs.astral.sh/uv/) and consulting the [tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for in-depth guidance.

```text
git checkout main

# Blender 4.2 or later

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/extensions/user_default/vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/extensions/user_default/vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\extensions\user_default\vrm" src\io_scene_vrm

# Blender 4.1.1 or earlier

# Linux
ln -Ts "$PWD/src/io_scene_vrm" "$HOME/.config/blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# macOS
ln -s "$PWD/src/io_scene_vrm" "$HOME/Library/Application Support/Blender/BLENDER_VERSION/scripts/addons/io_scene_vrm"
# Windows PowerShell
New-Item -ItemType Junction -Path "$Env:APPDATA\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" -Value "$(Get-Location)\src\io_scene_vrm"
# Windows Command Prompt
mklink /j "%APPDATA%\Blender Foundation\Blender\BLENDER_VERSION\scripts\addons\io_scene_vrm" src\io_scene_vrm
```