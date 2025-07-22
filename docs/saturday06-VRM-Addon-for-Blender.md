# VRM Add-on for Blender

**Enhance your Blender workflow with the VRM Add-on, a powerful tool for importing, exporting, and working with VRM models.** ([Original Repository](https://github.com/saturday06/VRM-Addon-for-Blender))

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Key Features

*   **VRM Import/Export:** Seamlessly import and export VRM models within Blender.
*   **VRM Humanoid Support:**  Add and configure VRM Humanoid rigs for easy animation.
*   **MToon Shader Integration:** Utilize the popular MToon shader for anime-style rendering.
*   **Physics-Based Material Support:** Create realistic materials with PBR workflows.
*   **Scripting API:** Automate tasks and extend functionality using Python scripts.

## Downloads

*   **Blender 4.2 or later:**  Get it from the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download it from the [🌐**Official Site**](https://vrm-addon-for-blender.info).

## Tutorials

| Installation                                                                                                       | Create Simple VRM                                                                                                       | Create Humanoid VRM                                                                                                          |
| :-------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics Based Material                                                                                        | Create Anime Style Material                                                                                                 | Automation with Python Scripts                                                                                                 |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                                                                                         | Development How-To                                                                                                           |                                                                                                                              |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a> |                                                                                                                              |

## Overview

This add-on integrates VRM-specific functionalities into Blender, streamlining the import, export, and manipulation of VRM models, including setting up MToon shaders and VRM Humanoid rigs. Development is ongoing, and contributions are welcome.

## Development

The main source code for the add-on resides in the `main` branch.  For efficient development, consider linking the `src/io_scene_vrm` folder within your Blender addons directory. For more advanced development tasks, leverage [astral.sh/uv](https://docs.astral.sh/uv/) for testing and building.  Refer to the [development tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for detailed instructions.

**Installation Instructions:**

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