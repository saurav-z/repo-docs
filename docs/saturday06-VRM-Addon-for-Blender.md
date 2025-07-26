# VRM Add-on for Blender: Create and Customize VRM Models Directly in Blender

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

This Blender add-on empowers you to import, export, and manipulate VRM models, streamlining your workflow for virtual avatars and 3D character creation. ([See the original repository](https://github.com/saturday06/VRM-Addon-for-Blender))

## Key Features

*   **VRM Import/Export:** Seamlessly import and export VRM files directly within Blender.
*   **VRM Humanoid Support:**  Add and configure VRM Humanoid rigs for animation and character control.
*   **MToon Shader Integration:** Apply and customize MToon shaders for a vibrant anime-style appearance.
*   **PBR Material Support:** Create realistic materials with physically based rendering.
*   **Animation Tools:** Animate your VRM models with dedicated tools.
*   **Scripting API:** Automate tasks and extend functionality with Python scripting.

## Installation

*   **For Blender 4.2 or later:**
    *   Install directly from the [Blender Extensions Platform](https://extensions.blender.org/add-ons/vrm).

*   **For Blender 2.93 to 4.1:**
    *   Download from the [Official Site](https://vrm-addon-for-blender.info).

## Tutorials

Explore these tutorials to get started with the VRM Add-on:

|                                          Installation                                           |                                      Create Simple VRM                                       |                                       Create Humanoid VRM                                        |
| :---------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
|                                     Create Physics Based Material                                      |                                        Create Anime Style Material                                         |                                         Automation with Python Scripts                                          |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> |     <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a>     |         <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a>         |
|                                        VRM Animation                                         |                                          Development How-To                                         |                                                                                                   |
|    <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |         <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>         |                                                                                                   |

## Development

The main source code resides in the `main` branch, specifically within the `src/io_scene_vrm` folder.  For efficient development, consider creating a symbolic link to this folder within your Blender add-ons directory.

Advanced development, including testing, can be facilitated with [astral.sh/uv](https://docs.astral.sh/uv/).  Refer to the [development tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for detailed instructions.

```bash
git checkout main

# Instructions for linking the source code to your Blender add-ons directory are provided in the original README.
```

## Contributing

Bug reports, feature requests, pull requests, and contributions are always welcome!