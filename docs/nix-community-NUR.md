# NUR: Your Community Hub for Nix Packages

**NUR (Nix User Repository) empowers you to discover and share Nix packages, built by the community, for the community.**  [View the NUR GitHub Repository](https://github.com/nix-community/NUR)

## Key Features:

*   **Community-Driven:** Access a vast collection of packages created and maintained by Nix users.
*   **Decentralized:** Share and install packages faster with a community-focused approach.
*   **User-Generated Content:** Explore packages not found in Nixpkgs, built from source.
*   **Automated Updates:** NUR automatically checks repositories and performs evaluation checks to ensure quality.
*   **Flexible Integration:** Seamlessly integrate NUR packages with Flakes, `packageOverrides`, NixOS, Home Manager, and devshells.
*   **Package Search:** Easily find packages through our dedicated search tool.

## Installation

### Using Flakes

Add the following to your `flake.nix`:

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

Add the following to `~/.config/nixpkgs/config.nix`:

```nix
{
  packageOverrides = pkgs: {
    nur = import (builtins.fetchTarball "https://github.com/nix-community/NUR/archive/main.tar.gz") {
      inherit pkgs;
    };
  };
}
```

For NixOS, add the following to `/etc/nixos/configuration.nix`
**Note:** If you want to use NUR in `nix-env`, `home-manager`, or `nix-shell`, you'll also need to include NUR in `~/.config/nixpkgs/config.nix`.

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

For a stable installation, pin the NUR version using `builtins.fetchTarball` with a specific commit hash.

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

**Important:** Review packages before installing them, as NUR doesn't regularly check for malicious content.

### Devshell Example (Flakes)

Add a single package from NUR to a devshell using the following `flake.nix`:

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
          overlays = [ nur.overlay ];
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

### NixOS Configuration (Flakes)

Integrate NUR modules and overlays into your NixOS configuration.

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
      # ...
      modules = [
        # Adds the NUR overlay
        nur.modules.nixos.default
        # NUR modules to import
        nur.legacyPackages."${system}".repos.iopq.modules.xraya
        # This adds the NUR nixpkgs overlay.
        # Example:
        # ({ pkgs, ... }: {
        #   environment.systemPackages = [ pkgs.nur.repos.mic92.hello-nur ];
        # })
      ];
    };
  };
}
```

### Home Manager Integration

Integrate with Home Manager by importing NUR modules into your `imports` attribute.

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
*   [nur-combined](https://github.com/nix-community/nur-combined/search) (for searching through all expressions)

## Adding Your Own Repository

1.  Create a repository with a `default.nix` file in the root.  Use the [repository template](https://github.com/nix-community/nur-packages-template) for a prepared structure.
2.  Use the `pkgs` argument from Nixpkgs when importing dependencies.
3.  Your repository should return a set of Nix derivations.

```nix
{ pkgs }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

  In this example, `hello-nur` is a directory containing a `default.nix`:

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

  # fails due to patch
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

4.  Build your packages using `nix-shell` or `nix-build`:

```console
$ nix-shell --arg pkgs 'import <nixpkgs> {}' -A hello-nur
nix-shell> hello
nix-shell> find $buildInputs
```

```console
$ nix-build --arg pkgs 'import <nixpkgs> {}' -A hello-nur
```
5.  For convenience, set a default value for the `pkgs` argument:

```nix
{ pkgs ? import <nixpkgs> {} }:
{
  hello-nur = pkgs.callPackage ./hello-nur {};
}
```

```console
$ nix-build -A hello-nur
```

6.  Add your repository to `repos.json`:

```console
$ git clone --depth 1 https://github.com/nix-community/NUR
$ cd NUR
```

7.  Edit `repos.json`:

```json
{
    "repos": {
        "mic92": {
            "url": "https://github.com/Mic92/nur-packages"
        },
        "<fill-your-repo-name>": {
            "url": "https://github.com/<your-user>/<your-repo>"
        }
    }
}
```

8.  Run `./bin/nur format-manifest` to sort the JSON alphabetically, then add and commit your changes. Do *not* commit `repos.json.lock`.
9.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a Different Root Expression

Set the `file` option in `repos.json` to load packages from a different file:

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

Use the [nur-update service](https://nur-update.nix-community.org/) to update NUR's lock file faster after pushing updates to your repository.  For example:

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Troubleshooting Updates

Evaluation errors can prevent updates.  Common issues include:

*   Incorrect `meta.license` attributes.
*   Using built-in fetchers instead of `pkgs.fetch*`.

Check the [latest build job](https://github.com/nix-community/NUR/actions) to see if your evaluation succeeded.

#### Local Evaluation Check

In your `nur-packages/` directory, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task.

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

Enable submodule support in `repos.json`:

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

Define modules, overlays, and library functions within your repository.

#### NixOS Modules

Place modules in the `modules` attribute:

```nix
{ pkgs }: {
  modules = import ./modules;
}
```

```nix
# modules/default.nix
{
  example-module = ./example-module.nix;
}
```

See example [here](https://github.com/Mic92/nur-packages/tree/master/modules).

Modules should be paths and not functions.  Modules with no `_class` are assumed to be NixOS and Home Manager modules. Use the `_class` attribute to specify `"nixos"` or `"home-manager"`.

#### Overlays

Place overlays in the `overlays` attribute:

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

#### Library Functions

Put reusable nix functions in the `lib` attribute:

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
        ## remote locations are also possible:
        # mic92 = import (builtins.fetchTarball "https://github.com/your-user/nur-packages/archive/main.tar.gz") { inherit pkgs; };
      };
    };
  };
}
```
### Overriding repositories with Flake

**Experimental** Note that flake support is still experimental and might change in future in a backwards incompatible way.

You can override repositories in two ways:

- With packageOverrides
```nix
{
  inputs.nur.url = "github:nix-community/NUR";
  inputs.paul.url = "path:/some_path/nur-paul"; # example: a local nur.repos.paul for development

  outputs = {self, nixpkgs, nur, paul }: {

  system = "x86_64-linux";

  nurpkgs = import nixpkgs { inherit system; };

  ...
  modules = [
       {
         nixpkgs.config.packageOverrides = pkgs: {
            nur = import nur {
              inherit pkgs nurpkgs;
              repoOverrides = { paul = import paul { inherit pkgs; }; };
            };
          };
        }
  ];
  ...
}
```
- With overlay
```nix
{
  modules = [
    {
      nixpkgs.overlays = [
        (final: prev: {
          nur = import nur {
            nurpkgs = prev;
            pkgs = prev;
            repoOverrides = { paul = import paul { pkgs = prev; }; };
          };
        })
      ];
    }
    ...
  ];
}
```

The repo must contain a `flake.nix` file in addition to a `default.nix`:  [flake.nix example](https://github.com/Mic92/nur-packages/blob/master/flake.nix)

## Contribution Guidelines

*   Ensure packages build and set the `meta.broken` attribute to `true` if they don't.
*   Provide meta attributes as described in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim.
*   Reuse packages from Nixpkgs where possible.

## Example Use Cases:

*   Packages for a limited audience.
*   Pre-release software.
*   Older package versions.
*   Automatically generated package sets.
*   Software with custom patches.
*   Experiments.

## Contact

Join the conversation in the matrix channel: [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) and the [NixOS Discourse](https://discourse.nixos.org/).