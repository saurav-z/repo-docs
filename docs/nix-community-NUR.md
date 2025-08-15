# Nix User Repository (NUR): Expand Your Nix Package Ecosystem

NUR is a community-driven repository that empowers Nix users to access and share packages, expanding the availability of software beyond [Nixpkgs](https://github.com/NixOS/nixpkgs/).

*   **Decentralized Package Sharing:** Easily install packages from user-maintained repositories, fostering a faster and more community-driven approach to package availability.
*   **Community-Driven:** Benefit from a wide array of packages contributed by fellow Nix enthusiasts, including pre-releases, niche software, and more.
*   **Automated Updates:** NUR automatically checks repositories and performs evaluation checks before updates, ensuring a stable and reliable experience.
*   **Flexible Installation:** Seamlessly integrate NUR into your Nix workflow using flakes, `packageOverrides`, or other methods.
*   **Extensible:** Supports NixOS modules, overlays, and library functions, allowing for comprehensive system customization.

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

Use either the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS, add to `/etc/nixos/configuration.nix`.  *Note: Include in `~/.config/nixpkgs/config.nix` for use in `nix-env`, home-manager, or `nix-shell`.*

### Pinning

Pin the version for faster builds and caching.

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages from the NUR namespace:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

*   **Important Security Note:**  NUR does not regularly check repositories for malicious content.  Always review expressions before installation.

### Using a Single Package in a Devshell

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, flake-utils, nur }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ nur.overlays.default ];
        };
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [ pkgs.nur.repos.mic92.hello-nur ];
        };
      }
    );
}
```

### Using the Flake in NixOS

```nix
{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    nur = {
      url = "github:nix-community/NUR";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs, nur }: {
    nixosConfigurations.myConfig = nixpkgs.lib.nixosSystem {
      modules = [
        nur.modules.nixos.default
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
      ];
    };
  };
}
```

### Integrating with Home Manager

```nix
let
  nur-no-pkgs = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {};
in
{
  imports = lib.attrValues nur-no-pkgs.repos.moredhel.hmModules.rawModules;

  services.unison = {
    enable = true;
    profiles = {
      org = {
        src = "/home/moredhel/org";
        dest = "/home/moredhel/org.backup";
        extraArgs = "-batch -watch -ui text -repeat 60 -fat";
      };
    };
  };
}
```

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file (template: [repository template](https://github.com/nix-community/nur-packages-template)).
2.  *Do not* import packages using `with import <nixpkgs> {}`. Instead, use the provided `pkgs` argument.
3.  Your repository should return a set of Nix derivations.

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

Example `hello-nur/default.nix`:

```nix
{ stdenv, fetchurl, lib }:

stdenv.mkDerivation rec {
  pname = "hello";
  version = "2.10";

  src = fetchurl {
    url = "mirror://gnu/hello/${pname}-${version}.tar.gz";
    sha256 = "0ssi1wpaf7plaswqqjwigppsg5fyh99vdlb9kzl7c9lng89ndq1i";
  };

  postPatch = ''
    sed -i -e 's/Hello, world!/Hello, NUR!/' src/hello.c
  '';

  doCheck = false;

  meta = with lib; {
    description = "A program that produces a familiar, friendly greeting";
    longDescription = ''
      GNU Hello is a program that prints "Hello, world!" when you run it.
      It is fully customizable.
    '';
    homepage = https://www.gnu.org/software/hello/manual/;
    changelog = "https://git.savannah.gnu.org/cgit/hello.git/plain/NEWS?h=v${version}";
    license = licenses.gpl3Plus;
    maintainers = [ maintainers.eelco ];
    platforms = platforms.all;
  };
}
```

Build your packages with `nix-shell` or `nix-build`:

```console
$ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```

```console
$ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```

For development, set a default `pkgs` argument:

```nix
{ pkgs ? import <nixpkgs> {} }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

```console
$ nix-build -A hello-nur
```

Add your repository to `repos.json`:

```console
$ git clone --depth 1 https://github.com/nix-community/NUR
$ cd NUR
```

Edit `repos.json`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages"
        },
        "<your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

Run `./bin/nur format-manifest` and `git add repos.json`, *not* `repos.json.lock`. Commit and push, then open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Use a Different Nix File as Root Expression

Set the `file` option in `repos.json`:

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

### Updating NUR's Lock File

Use https://nur-update.nix-community.org/update?repo=<your_repo_name> after pushing updates to your repository. More details at [nur-update documentation](https://github.com/nix-community/nur-update#nur-update-endpoint).

### Why Are My NUR Packages Not Updating?

Check for evaluation errors, which can prevent repository updates.  Common causes:

*   Incorrect license attribute.
*   Using a builtin fetcher.  Use `pkgs.fetch*` instead.

Check the [latest build job](https://github.com/nix-community/NUR/actions) for evaluation results.

#### Local Evaluation Check

In your `nur-packages/` folder, run the evaluation check task:

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

### Git Submodules

Enable submodules in `repos.json`:

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

Organize these in namespaces within your repository for discoverability.

#### Providing NixOS Modules

Place modules in the `modules` attribute:

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

Example:

```nix
# modules/default.nix
{
  example-module = ./example-module.nix;
}
```

Modules should be paths, not functions.  A module with no `_class` is assumed to be both a NixOS and Home Manager module.  Set `_class` to `"nixos"` or `"home-manager"` for specific module types.

#### Providing Overlays

Use the `overlays` attribute:

```nix
# default.nix
{
  overlays = {
    hello-overlay = import ./hello-overlay;
  };
}
```

```nix
# hello-overlay/default.nix
self: super: {
  hello = super.hello.overrideAttrs (old: {
    separateDebugInfo = true;
  });
}
```

#### Providing Library Functions

Place reusable functions in the `lib` attribute:

```nix
{ pkgs }:
with pkgs.lib;
{
  lib = {
    hexint = x: hexvals.${toLower x};

    hexvals = listToAttrs (imap (i: c: { name = c; value = i - 1; })
      (stringToCharacters "0123456789abcdef"));
  };
}
```

## Overriding Repositories

Override repositories using the `repoOverrides` argument:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
      repoOverrides = {
        mic92 = import ../nur-packages { inherit pkgs; };
      };
    };
  };
}
```

### Overriding Repositories with Flake

**Experimental:**  See examples in the original README.

## Contribution Guidelines

*   Packages must build and use the `meta.broken` attribute if broken.
*   Use standard meta attributes as described in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories lean.
*   Reuse packages from Nixpkgs where possible.

## Examples

*   Packages for a small audience.
*   Pre-releases.
*   Old versions of packages.
*   Automatic package sets.
*   Software with custom patches.
*   Experiments.

## Contact

Join the conversation on [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or [https://discourse.nixos.org](https://discourse.nixos.org/).

[Back to Top](#top)