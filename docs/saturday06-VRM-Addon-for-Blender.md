# VRM Add-on for Blender: Create and Customize VRM Models Directly in Blender

Easily import, export, and modify VRM models within Blender using the **VRM Add-on for Blender**. ([Original Repo](https://github.com/saturday06/VRM-Addon-for-Blender))

[![CI status](https://github.com/saturday06/VRM-Addon-for-Blender/actions/workflows/test.yml/badge.svg?branch=main)](https://github.com/saturday06/VRM-Addon-for-Blender/actions) [![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)

## Key Features

*   **VRM Import/Export:** Seamlessly bring VRM models into Blender and export your creations in the VRM format.
*   **Humanoid Rigging:**  Add and configure VRM Humanoid rigs within Blender.
*   **MToon Shader Support:** Implement and customize MToon shaders for anime-style rendering.
*   **PBR Material Support:**  Create realistic, physics-based materials for your VRM models.
*   **Animation Tools:** Animate your VRM models directly within Blender.
*   **Scripting API:** Automate tasks and customize your workflow with the Python scripting API.

## Download

Choose the version compatible with your Blender installation:

*   **Blender 4.2 or later:** Download from the [🛠️**Blender Extensions Platform**](https://extensions.blender.org/add-ons/vrm).
*   **Blender 2.93 to 4.1:** Download from the [🌐**Official Website**](https://vrm-addon-for-blender.info).

## Tutorials

Explore these tutorials to get started:

| Installation                                                                                                                                   | Create Simple VRM                                                                                                                                    | Create Humanoid VRM                                                                                                                                    |
| :---------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: | :-----------------------------------------------------------------------------------------------------------------------------------------------------: |
| <a href="https://vrm-addon-for-blender.info/en/installation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/installation.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-simple-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/simple.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/create-humanoid-vrm-from-scratch?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/humanoid.gif"></a> |
| Create Physics Based Material                                                                                                               | Create Anime Style Material                                                                                                                          | Automation with Python Scripts                                                                                                                         |
| <a href="https://vrm-addon-for-blender.info/en/material-pbr?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_pbr.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/material-mtoon?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/material_mtoon.gif"></a> | <a href="https://vrm-addon-for-blender.info/en/scripting-api?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/scripting_api.gif"></a> |
| VRM Animation                                                                                                                                | Development How-To                                                                                                                                     |                                                                                                                                                         |
| <a href="https://vrm-addon-for-blender.info/en/animation?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    | <a href="https://vrm-addon-for-blender.info/en/development?locale_redirection"><img width="200" src="https://vrm-addon-for-blender.info/images/animation.gif"></a>    |                                                                                                                                                         |

## Development

The core code is in the [`main`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main) branch, specifically within the [`src/io_scene_vrm`](https://github.com/saturday06/VRM-Addon-for-Blender/tree/main/src/io_scene_vrm) directory.

For advanced development, use [astral.sh/uv](https://docs.astral.sh/uv/).  Refer to the [tutorial](https://vrm-addon-for-blender.info/en/development?locale_redirection) for detailed instructions.