# MAVLink: Lightweight Communication for Drones and Ground Stations

**MAVLink empowers seamless communication between drones and ground control stations with a lightweight and efficient message library.** Developed by the MAVLink community ([Original Repo](https://github.com/mavlink/mavlink)), this library provides a standardized way for various components to interact, especially in resource-constrained environments.

## Key Features

*   **Lightweight & Efficient:** Designed for minimal overhead, ideal for systems with limited bandwidth, RAM, and flash memory.
*   **Header-Only Library:** Simplifies integration and reduces dependencies.
*   **Cross-Platform Support:** Compatible with multiple programming languages, including C, Python, and more (see [Supported Languages](https://mavlink.io/en/#supported_languages)).
*   **XML-Based Message Definitions:** Uses XML files to define message sets ("dialects"), promoting interoperability.
*   **Field-Proven:** Deployed in numerous products, ensuring reliability and stability.
*   **Optimized for Resource-Constrained Systems:** The C implementation is specifically optimized for embedded systems.

## Quick Start Guide: Generating C Headers

Here's how to quickly get started generating C headers for MAVLink:

1.  **Install Dependencies:**
    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**
    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Dependencies:**
    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**  (Example for the common dialect, version 2.0)
    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

## Integrating with CMake

1.  **Install Headers Locally:**
    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Use `find_package` in `CMakeLists.txt`:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Pass the install directory to cmake:**
    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    Refer to the [examples/c](examples/c) directory for a complete example.

*Note: MAVLink is a header-only library, even though cmake `target_link_libraries` is used*

## Further Resources

*   **Documentation:** [mavlink.io/en/](https://mavlink.io/en/)
*   **Support:** [mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [mavlink.io/en/#license](https://mavlink.io/en/#license)
*   **Using C MAVLink Libraries (mavgen):** [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/)
*   **Installing MAVLink Toolchain:** [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html)
*   **Generating MAVLink Libraries:** [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html)
*   **Using MAVLink Libraries:** [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html)