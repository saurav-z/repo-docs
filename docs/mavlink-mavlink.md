# MAVLink: The Lightweight Communication Protocol for Drones and Ground Control Stations

MAVLink is a powerful, header-only message library that enables seamless communication between drones, ground control stations, and other components in the unmanned aerial vehicle (UAV) ecosystem. Find the original repo [here](https://github.com/mavlink/mavlink).

## Key Features of MAVLink

*   **Lightweight and Efficient:** Designed for applications with limited bandwidth and resource-constrained systems.
*   **Header-Only Library:** Simplifies integration and reduces dependencies.
*   **Cross-Platform Support:** Compatible with numerous programming languages.
*   **Message-Set Specifications:** Defined in XML files for different systems ("dialects").
*   **Python-Based Tools:** Generate source code for supported languages and provide utilities.
*   **Field-Proven:** Deployed in numerous products as an interoperability interface.
*   **Highly Optimized:** C reference implementation is optimized for resource-constrained systems.

## Getting Started: Generate C Headers

Follow these steps to set up a minimal MAVLink environment on Ubuntu LTS 20.04 or 22.04:

1.  **Install Dependencies:**

    ```bash
    sudo apt install python3-pip
    ```

2.  **Clone the Repository:**

    ```bash
    git clone https://github.com/mavlink/mavlink.git --recursive
    cd mavlink
    ```

3.  **Install Python Requirements:**

    ```bash
    python3 -m pip install -r pymavlink/requirements.txt
    ```

4.  **Generate C Headers:**

    ```bash
    python3 -m pymavlink.tools.mavgen --lang=C --wire-protocol=2.0 --output=generated/include/mavlink/v2.0 message_definitions/v1.0/common.xml
    ```

## Using MAVLink with CMake

1.  **Install Headers Locally:**

    ```bash
    cmake -Bbuild -H. -DCMAKE_INSTALL_PREFIX=install -DMAVLINK_DIALECT=common -DMAVLINK_VERSION=2.0
    cmake --build build --target install
    ```

2.  **Integrate into CMakeLists.txt:**

    ```cmake
    find_package(MAVLink REQUIRED)
    add_executable(my_program my_program.c)
    target_link_libraries(my_program PRIVATE MAVLink::mavlink)
    ```

3.  **Configure with CMake:**

    ```bash
    cd ../my_program
    cmake -Bbuild -H. -DCMAKE_PREFIX_PATH=../mavlink/install
    ```

    See the [examples/c](examples/c) directory for a complete example.

    *Note:  While `target_link_libraries` is used, MAVLink is header-only, so it doesn't actually "link".*

## Further Instructions and Resources

*   Refer to [Using C MAVLink Libraries (mavgen)](https://mavlink.io/en/mavgen_c/) for comprehensive C library usage.
*   Install MAVLink on other Ubuntu platforms and Windows using the instructions in [Installing the MAVLink Toolchain](https://mavlink.io/en/getting_started/installation.html).
*   Learn how to generate MAVLink libraries for other programming languages in [Generating MAVLink Libraries](https://mavlink.io/en/getting_started/generate_libraries.html).
*   Explore how to leverage the generated libraries in your projects in [Using MAVLink Libraries](https://mavlink.io/en/getting_started/use_libraries.html).

## Important Links

*   **Documentation/Website:** [mavlink.io/en/](https://mavlink.io/en/)
*   **Discussion/Support:** [mavlink.io/en/#support](https://mavlink.io/en/#support)
*   **Contributing:** [mavlink.io/en/contributing/contributing.html](https://mavlink.io/en/contributing/contributing.html)
*   **License:** [mavlink.io/en/#license](https://mavlink.io/en/#license)