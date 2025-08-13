# NUR: The Nix User Repository

**Expand your Nix package ecosystem with NUR, a community-driven repository for user-contributed packages, modules, and more.**

[![GitHub Repo stars](https://img.shields.io/github/stars/nix-community/NUR?style=social)](https://github.com/nix-community/NUR)

Nix User Repository (NUR) is a community-driven meta-repository that extends the Nix package manager, offering a decentralized way to access and share Nix packages, modules, and other Nix expressions. Unlike Nixpkgs, packages in NUR are built from source and are **not** reviewed by Nixpkgs members, fostering rapid community contributions and experimentation.

**Key Features:**

*   **Community-Driven:** Access packages and modules contributed by the Nix community.
*   **Decentralized:** Share and discover packages faster than through traditional channels.
*   **Flexible:** Supports packages, NixOS modules, Home Manager modules, and library functions.
*   **Automated Checks:** Evaluates repositories and propagates updates automatically.
*   **Easy Integration:** Seamlessly integrate with Flakes, `packageOverrides`, Home Manager, and NixOS configurations.
*   **Package Search:** Find packages using the [NUR Packages search](https://nur.nix-community.org/) or explore the [nur-combined repository](https://github.com/nix-community/nur-combined).

## Installation

### Using Flakes

Include NUR in your `flake.nix`:

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>` after.

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

### Pinning (Recommended for Stability)

Pin the NUR version for reliable builds:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/<commit-hash>.tar.gz";
  sha256 = "<sha256-hash>";
}
```
**Note:** You can find the commit hash from the [NUR commits page](https://github.com/nix-community/NUR/commits/main) and generate the SHA256 hash using `nix-prefetch-url --unpack <url>`.

## How to Use

Install and use packages from NUR:

```bash
$ nix-shell -p nur.repos.mic92.hello-nur
# OR
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
# OR (NixOS)
environment.systemPackages = with pkgs; [ nur.repos.mic92.hello-nur ];
```

**Important Note:**  *Exercise caution and review expressions before installing packages from NUR.*

## Adding Your Own Repository

1.  **Create a Repository:** Structure your repository with a `default.nix` file at the root.
2.  **Package Definition:** Define packages using a function that takes `pkgs` as an argument.
3.  **Test:** Use `nix-shell` or `nix-build` to test your packages.
4.  **Add to NUR:**
    *   Clone the NUR repository: `git clone --depth 1 https://github.com/nix-community/NUR`
    *   Edit `repos.json` to include your repository's URL.
    *   Run `./bin/nur format-manifest` to format the `repos.json`
    *   Commit and push your changes.
    *   Open a pull request to the NUR repository.

### Using a different root nix file

To use a different file instead of `default.nix` to load packages from, set the `file` option to a path relative to the repository root.

### Update NUR's lock file after updating your repository

Use the [nur-update service](https://nur-update.nix-community.org/update?repo=<your-repo-name>) to speed up lock file updates.

### Why are my NUR packages not updating?

* Make sure your evaluation doesn't contain any errors.
* Common errors include wrong license attributes, or using builtin fetchers.
* You can find out if your evaluation succeeded by checking the [latest build job](https://github.com/nix-community/NUR/actions).

#### Local evaluation check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task

```sh
nix-env -f . -qa \* --meta \
  --allowed-uris https://static.rust-lang.org \
  --option restrict-eval true \
  --option allow-import-from-derivation true \
  --drv-path --show-trace \
  -I nixpkgs=$(nix-instantiate --find-file nixpkgs) \
  -I ./ \
  --json | jq -r 'values | .[].name'
```

## Advanced Features

*   **Git Submodules:** Enable Git submodules by setting `submodules = true` in `repos.json`.
*   **NixOS Modules, Overlays, and Library Functions:** Organize your modules, overlays, and library functions within your repository using the `modules`, `overlays`, and `lib` attributes.
*   **Overriding Repositories:** Use the `repoOverrides` argument to test changes before publishing.  Also supports flake based repo overrides.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not buildable.
*   Use standard Nixpkgs meta attributes.
*   Keep repositories lightweight.
*   Reuse packages from Nixpkgs when possible.

## Examples of Packages Well-Suited for NUR

*   Packages for a small audience
*   Pre-release versions
*   Old versions of packages
*   Automatically generated packages
*   Software with opinionated patches
*   Experiments

## Contact

Join the conversation:
*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org/](https://discourse.nixos.org/)

[Visit the NUR GitHub Repository](https://github.com/nix-community/NUR) for more information.