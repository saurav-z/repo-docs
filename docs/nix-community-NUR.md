# NUR: The Nix User Repository

**Expand your Nix ecosystem with NUR, a community-driven repository for user-contributed packages and configurations.** Explore and install packages faster than ever!

[Back to the Original Repository](https://github.com/nix-community/NUR)

**Key Features:**

*   **Community-Driven:** Access a wide variety of packages and configurations contributed by the Nix community.
*   **Decentralized:** Get access to packages quickly without waiting for Nixpkgs review.
*   **Easy Installation:** Integrate NUR into your Nix setup using flakes or `packageOverrides`.
*   **Automatic Updates:**  NUR automatically checks and validates repositories for updates.
*   **Flexible Use:** Install packages via `nix-shell`, `nix-env`, or in your `configuration.nix`.
*   **NixOS Modules and Overlays:**  Discover and use NixOS modules, overlays, and library functions from community repositories.
*   **Package Search:** Easily find packages through [Packages search for NUR](https://nur.nix-community.org/) or [nur-combined](https://github.com/nix-community/nur-combined).

**Installation:**

*   **Using Flakes:**

    ```nix
    {
      inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
        nur = {
          url = "github:nix-community/NUR";
          inputs.nixpkgs.follows = "nixpkgs";
        };
      };
    };
    ```

    Then, use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

*   **Using `packageOverrides`:**

    1.  Add NUR to your `~/.config/nixpkgs/config.nix`:

        ```nix
        {
          packageOverrides = pkgs: {
            nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
              inherit pkgs;
            };
          };
        }
        ```

    2.  For NixOS, add to your `/etc/nixos/configuration.nix` (also required for `nix-env`, home-manager or `nix-shell`):

        ```nix
        {
          nixpkgs.config.packageOverrides = pkgs: {
            nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
              inherit pkgs;
            };
          };
        }
        ```

*   **Pinning:**  Pin specific NUR versions for reproducibility.  Use `nix-prefetch-url` to generate the `sha256` hash.

    ```nix
    builtins.fetchTarball {
      url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
      sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
    }
    ```

**How to Use:**

Install packages from NUR using:

*   `nix-shell -p nur.repos.mic92.hello-nur`
*   `nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur`
*   In `configuration.nix`:

    ```nix
    environment.systemPackages = with pkgs; [
      nur.repos.mic92.hello-nur
    ];
    ```

**Contributing Your Own Repository:**

1.  Create a Git repository with a `default.nix` file (use the provided [template](https://github.com/nix-community/nur-packages-template)).
2.  In your `default.nix`, take all dependencies from Nixpkgs from the given `pkgs` argument.
3.  Structure your repository:  packages, NixOS modules, overlays, and library functions.
4.  Add your repository details to NUR's `repos.json` and submit a pull request.
5.  Update the NUR lock file by using [nur-update service](https://nur-update.nix-community.org/update).

**Important Considerations:**

*   **Security:** Review packages before installation; NUR doesn't perform regular security checks.
*   **Evaluation Errors:** Ensure your repository evaluates successfully during NUR's update process.  Fix any errors related to metadata, builtin fetchers, etc.

**Overriding Repositories:**

Override repositories to test changes before publishing or modify existing packages. Using `repoOverrides` or Flakes.

**Contact:**

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)