# NUR: Your Community Hub for Nix Packages

**Nix User Repository (NUR) is a community-driven meta-repository that expands the Nix ecosystem by providing access to user-contributed packages, offering a faster and more decentralized way to discover and install software.** Learn more on the [original repository](https://github.com/nix-community/NUR).

**Key Features:**

*   📦 **Community-Driven:** Access a wide range of user-contributed Nix packages.
*   🚀 **Fast Package Discovery:** Get access to new packages quicker than through Nixpkgs.
*   🔄 **Automated Updates:** Benefit from automatic checks and evaluation before updates.
*   🛠️ **Flexible Installation:** Supports flakes, packageOverrides, and NixOS configurations.
*   🤝 **Community-Focused:** Designed for sharing packages that may not be in Nixpkgs.

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

Then, use the overlay (`overlays.default`) or `legacyPackages.<system>`.

### Using `packageOverrides`

Add this to `~/.config/nixpkgs/config.nix`:

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

Pinning the NUR version is recommended to avoid potential build issues due to download caching:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using `nix-shell`, `nix-env`, or within your NixOS configuration.

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

or

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

or

```console
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

**Important Note:** NUR does not regularly check for malicious content. Always review expressions before installation.

## Integrating with your workflows

*   **Devshell:** Use a single package in a devshell for local development.
*   **NixOS:** Integrate NUR using overlays and modules within your NixOS configuration.
*   **Home Manager:** Easily integrate NUR with Home Manager by adding modules to the `imports` attribute.

## Finding Packages

*   [Packages search for NUR](https://nur.nix-community.org/)
*   [nur-combined](https://github.com/nix-community/nur-combined/search)

## Contributing Your Own Repository

1.  **Create a Repository:** Make a repository with a `default.nix`.  The [repository template](https://github.com/nix-community/nur-packages-template) provides a prepared structure.
2.  **Dependencies:** Take all dependencies from Nixpkgs from the given `pkgs` argument.
3.  **Nix Derivations:** Your repository should return a set of Nix derivations.
4.  **Add to `repos.json`:** Update the `repos.json` file in the NUR repository.
5.  **Submit a Pull Request:** Open a pull request to add your repository to NUR.

### Using a different nix file as root expression

Set the `file` option in `repos.json`

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

Use [nur-update.nix-community.org](https://nur-update.nix-community.org/)

```console
curl -XPOST https://nur-update.nix-community.org/update?repo=mic92
```

### Why are my NUR packages not updating?

Check the [latest build job](https://github.com/nix-community/NUR/actions) for errors. Common causes: wrong licenses, or usage of builtin fetchers.

#### Local evaluation check

Run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task in your `nur-packages/` folder

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

### Git submodules

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

### NixOS modules, overlays and library function support

*   **NixOS modules:** Place them in the `modules` attribute.
*   **Overlays:** Use the `overlays` attribute.
*   **Library functions:** Put reusable functions in the `lib` attribute.

## Overriding repositories

### Overriding repositories using `repoOverrides`

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

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if broken.
*   Supply meta attributes as described in the [Nixpkgs manual](https://nixos.org/nixpkgs/manual/#sec-standard-meta-attributes).
*   Keep repositories slim; reuse Nixpkgs packages when applicable.

## Examples of packages suitable for NUR:

*   Niche packages
*   Pre-releases
*   Older versions of packages
*   Generated package sets
*   Software with custom patches
*   Experiments

## Contact

Join us on matrix [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) and [https://discourse.nixos.org](https://discourse.nixos.org/).