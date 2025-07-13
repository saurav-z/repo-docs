# NUR: The Nix User Repository

**Expand your Nix ecosystem with community-driven packages, modules, and overlays using the Nix User Repository, a decentralized collection of user-contributed resources.  Check out the original repo at [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).**

## Key Features

*   **Community-Driven:** Access a wide array of packages, modules, and overlays contributed by the Nix community.
*   **Decentralized:** Discover and install packages not found in Nixpkgs, built from source and without Nixpkgs review.
*   **Easy Integration:** Seamlessly integrate NUR into your NixOS configurations, Home Manager setups, and devshells.
*   **Flake Support:** Utilize NUR's overlay and legacyPackages in your Nix flakes.
*   **Automatic Updates:** Benefit from automatic repository checks and evaluation checks.
*   **Package Discovery:** Easily find packages using the provided search tools.
*   **Customization:** Configure and override repositories to test changes or incorporate personal modifications.

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

Then, utilize the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add NUR to your `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to `/etc/nixos/configuration.nix`:

```nix
{
  nixpkgs.config.packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

### Pinning

Pin the NUR version for stability:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages with:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or in `configuration.nix`:

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important:** *NUR is community-driven.  Always review expressions before installing.*

### Examples
*   [Using a single package in a devshell](https://github.com/nix-community/NUR#using-a-single-package-in-a-devshell)
*   [Using the flake in NixOS](https://github.com/nix-community/NUR#using-the-flake-in-nixos)
*   [Integrating with Home Manager](https://github.com/nix-community/NUR#integrating-with-home-manager)

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing Your Own Repository

1.  Create a repository with a `default.nix`.
2.  DO NOT import packages for example `with import <nixpkgs> {};`. Instead take all dependency you want to import from Nixpkgs from the given `pkgs` argument.
3.  Your repository should return a set of Nix derivations.
4.  Add your repo to NUR's `repos.json`.
5.  Submit a pull request.

### Using a different nix file as root expression

Set the `file` option to a path relative to the repository root:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "file": "subdirectory/default.nix"
        }
    }
}
```

### Update NUR's lock file after updating your repository

Use our service to update NUR faster:
```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```
Check out the [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

### Why are my NUR packages not updating?

Typical evaluation errors include:
*   Using a wrong license attribute in the metadata.
*   Using a builtin fetcher because it will cause access to external URLs during evaluation. Use pkgs.fetch* instead (i.e. instead of `builtins.fetchGit` use `pkgs.fetchgit`)
You can find out if your evaluation succeeded by checking the [latest build job](https://github.com/nix-community/NUR/actions).

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

On success, this shows a list of your packages

### Git submodules

To fetch git submodules in repositories set `submodules`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages",
            "submodules": true
        }
    }
}
```

### NixOS Modules, Overlays, and Library Functions

Organize NixOS modules, overlays, and library functions in your repository using the following attributes:

*   `modules`: For NixOS modules.
*   `overlays`: For overlays.
*   `lib`: For library functions.

[Examples](https://github.com/nix-community/NUR#nixos-modules-overlays-and-library-function-support) are provided in the original README.

## Overriding Repositories

Override repositories using the `repoOverrides` argument.

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```

### Overriding repositories with Flake

See the original README for details on overriding repos with Flakes.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken` if not.
*   Use standard `meta` attributes.
*   Keep repositories slim.
*   Reuse packages from Nixpkgs when applicable.

## Examples of Packages in NUR

*   Packages for a small audience.
*   Pre-releases.
*   Legacy package versions.
*   Automatic package generation.
*   Software with custom patches.
*   Experiments.

## Contact

*   Matrix: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org)
*   Discourse: [https://discourse.nixos.org](https://discourse.nixos.org/)