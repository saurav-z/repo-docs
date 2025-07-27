# NUR: Your Community Hub for Nix Packages

**Nix User Repository (NUR) empowers the Nix community to share and install packages faster, independently from Nixpkgs.** [Learn More](https://github.com/nix-community/NUR)

## Key Features

*   **Decentralized Package Sharing:**  Access a wide range of user-contributed packages, expanding your Nix ecosystem.
*   **Community-Driven:** Benefit from packages created and maintained by fellow Nix users.
*   **Faster Package Availability:**  Get access to new packages and pre-releases quicker than through Nixpkgs.
*   **Automatic Updates:** NUR automatically checks and evaluates repository updates.
*   **Flake and Package Override Support:** Easy integration with Flakes and `packageOverrides` for flexible package management.
*   **NixOS Modules, Overlays and Library Function Support:** Share and use NixOS modules, overlays and library functions in a discoverable way.

## Installation

### Using Flakes

Integrate NUR into your `flake.nix`:

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

Then use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `~/.config/nixpkgs/config.nix` or `/etc/nixos/configuration.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

**Pinning:** For a stable installation, pin the NUR version using `fetchTarball` with a specific SHA256 hash.

## How to Use

Install packages from the NUR namespace using `nix-shell`, `nix-env`, or `configuration.nix`:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

## Examples

*   **Devshell with a Single Package:** Easily integrate NUR packages into your development shells.
*   **Using NUR in NixOS:** Leverage NUR modules and overlays within your NixOS configuration.
*   **Integrating with Home Manager:** Seamlessly incorporate NUR packages into your Home Manager setup.

## Finding Packages

*   **Package Search:** Explore available packages through [Packages search for NUR](https://nur.nix-community.org/) or search our [nur-combined](https://github.com/nix-community/nur-combined) repository.

## Contributing Your Own Repository

1.  Create a repository with a `default.nix`. The [repository template](https://github.com/nix-community/nur-packages-template) provides a good starting point.
2.  Ensure packages take dependencies from the `pkgs` argument provided.
3.  Add your repository information to the `repos.json` file in the NUR repository.
4.  Submit a pull request.

**Important:**  Regularly check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation success and check your repository build using the provided task.  Ensure repositories are buildable on Nixpkgs unstable.

## Additional Features

*   **Git Submodules:** Enable git submodules using the `submodules` setting in `repos.json`.
*   **Overriding Repositories:** Test changes before publishing using the `repoOverrides` argument with `packageOverrides` or with Flakes.
*   **Using different nix files as root expression:**  Set the `file` option to load packages from a specific file.
*   **NixOS modules, overlays and library function support:** Organize and share modules, overlays, and library functions.

## Contribution Guidelines

*   Build packages successfully and set the `meta.broken` attribute appropriately.
*   Supply the relevant metadata attributes, per the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean.
*   Leverage packages from Nixpkgs when possible.

## Contact

Join the conversation on our Matrix channel at [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or on [https://discourse.nixos.org](https://discourse.nixos.org/).