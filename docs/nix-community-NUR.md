# Nix User Repository (NUR): Community-Driven Packages for NixOS

**NUR expands NixOS with user-contributed packages, modules, and more, offering rapid access to community-built software.**

[View the original repository on GitHub](https://github.com/nix-community/NUR)

## Key Features

*   **Decentralized Package Sharing:** Quickly access and install packages contributed by the Nix community, outside of the standard Nixpkgs.
*   **Community-Driven:** Benefit from a wide range of user-created packages, modules, and more.
*   **Fast Updates:** Get access to new software and experimental packages without waiting for Nixpkgs updates.
*   **Flexible Installation:** Integrate NUR with flakes, `packageOverrides`, and NixOS configurations.
*   **Extensible:** Supports NixOS modules, overlays, and library functions, offering enhanced customization.
*   **Automatic Evaluation Checks:** Ensures the integrity of package definitions before updates, to maintain stability.

## Installation

### Using Flakes

Add NUR to your `flake.nix`:

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

Then use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add the following to your `~/.config/nixpkgs/config.nix` (for login user) or `/etc/nixos/configuration.nix` (for NixOS):

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

**Pinning:**  For stable builds, pin the NUR version using a specific commit hash (see the original README for detailed instructions).

## How to Use

Install or use packages from the NUR namespace:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```bash
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Security Note:**  *Always review packages before installing them.* NUR packages are not reviewed by Nixpkgs maintainers.

### Using a Single Package in a Devshell

See the original README for a `flake.nix` example.

### Integrating with Home Manager

See the original README for an example.

## Finding Packages

*   **Packages Search for NUR:** [https://nur.nix-community.org/](https://nur.nix-community.org/)
*   **nur-combined Repository:** [https://github.com/nix-community/nur-combined/search](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file.  Use the [repository template](https://github.com/nix-community/nur-packages-template) as a starting point.
2.  Define packages using the `pkgs` argument from Nixpkgs.  *Do not* import packages directly from `<nixpkgs>`.
3.  Structure your packages following the example provided in the original README.
4.  Add your repository information to `repos.json` (in the NUR repo).
5.  Run `./bin/nur format-manifest`, then add, commit, and push your changes to the NUR repository.
6.  Submit a pull request to the [NUR repository](https://github.com/nix-community/NUR).

**Important:**
* Each URL must point to a Git repository.
*  Ensure your repository is buildable on Nixpkgs unstable.
*  Use the [nur-update service](https://nur-update.nix-community.org/update?repo=mic92) to update NUR faster after pushing changes.
*  Check the [latest build job](https://github.com/nix-community/NUR/actions) to verify successful evaluation.

### Using a Different Root Expression
Use the `file` option in `repos.json`.

### Git Submodules
Set `submodules: true` in `repos.json`

### NixOS Modules, Overlays, and Library Function Support
Place these in their own namespaces within your repository ( `modules`, `overlays`, `lib` attributes). Refer to the original README for examples.

### Overriding Repositories

Use `repoOverrides` with `packageOverrides` or through flakes (refer to the original README for code samples).

## Contribution Guidelines

*   Ensure packages build and set the `meta.broken` attribute if necessary.
*   Provide standard `meta` attributes as described in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean.
*   Reuse packages from Nixpkgs when possible.

## Examples of suitable packages

Refer to the original README for a list of suggested package types for the NUR.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)