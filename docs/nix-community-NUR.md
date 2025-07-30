# Nix User Repository (NUR): Your Gateway to Community-Driven Nix Packages

**NUR expands Nix package availability with user-contributed repositories, offering a more decentralized and rapid way to access cutting-edge software.** ([Back to Original Repo](https://github.com/nix-community/NUR))

## Key Features:

*   **Community-Driven:** Access packages created and maintained by the Nix community.
*   **Rapid Updates:**  Get access to new packages and updates more quickly than through Nixpkgs.
*   **Flexible Installation:** Integrates seamlessly with flakes, packageOverrides, NixOS configurations, and Home Manager.
*   **Automated Checks:** NUR automatically validates repositories and performs evaluation checks to ensure quality.
*   **Package Discovery:** Find packages easily using the [Packages search for NUR](https://nur.nix-community.org/) or [nur-combined](https://github.com/nix-community/nur-combined).
*   **Extensible:** Supports NixOS modules, overlays, and library functions.

## Installation

### Using Flakes

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

Use the overlay (`overlays.default`) or `legacyPackages.<system>`.

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

For NixOS, add to your `/etc/nixos/configuration.nix`.
*Important*: If using NUR in `nix-env`, `home-manager`, or `nix-shell`, also add to `~/.config/nixpkgs/config.nix`.

### Pinning

Pinning improves build reliability by specifying a `sha256` hash:

```nix
builtins.fetchTarball {
  url = "https://github.com/nix-community/NUR/archive/3a6a6f4da737da41e27922ce2cfacf68a109ebce.tar.gz";
  sha256 = "04387gzgl8y555b3lkz9aiw9xsldfg4zmzp930m62qw8zbrvrshd";
}
```

## How to Use

Install packages using:

```console
$ nix-shell -p nur.repos.mic92.hello-nur
nix-shell> hello
Hello, NUR!
```

Or:

```console
$ nix-env -f '<nixpkgs>' -iA nur.repos.mic92.hello-nur
```

Or (NixOS):

```nix
# configuration.nix
environment.systemPackages = with pkgs; [
  nur.repos.mic92.hello-nur
];
```

***Always review expressions before installing.***

## Adding Your Own Repository

1.  Create a repository with a `default.nix` (see [template](https://github.com/nix-community/nur-packages-template)).
2.  Avoid `with import <nixpkgs> {}`. Use the `pkgs` argument.
3.  Define packages as a set of Nix derivations.
4.  Test with `nix-build` or `nix-shell`.
5.  Add your repo to `repos.json`.
6.  Run `./bin/nur format-manifest`, commit, and push.
7.  Open a pull request to [https://github.com/nix-community/NUR](https://github.com/nix-community/NUR).

### Using a different nix file as root expression

To use a different file instead of `default.nix` to load packages from, set the `file`
option to a path relative to the repository root. See examples above.

### Update NUR's lock file after updating your repository

Use our service at https://nur-update.nix-community.org/ to update NUR faster after your updates. See [github page](https://github.com/nix-community/nur-update#nur-update-endpoint) for further details

### Troubleshooting Package Updates

Failed evaluations can stop package updates.  Check the [latest build job](https://github.com/nix-community/NUR/actions) and ensure:

*   Correct license attributes in metadata.
*   Use `pkgs.fetch*` instead of `builtins.fetch*` during evaluation.

#### Local Evaluation Check

In your `nur-packages/` folder, run the [check evaluation](https://github.com/nix-community/nur-packages-template/blob/main/.github/workflows/build.yml) task
and ensure all packages builds
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

Enable submodules by setting `submodules: true` in `repos.json`.

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

*   **Modules:**  Place in the `modules` attribute (see [example](https://github.com/Mic92/nur-packages/tree/master/modules)).
*   **Overlays:** Place in the `overlays` attribute.
*   **Library Functions:** Place reusable functions in the `lib` attribute.

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
### Overriding Repositories with Flake

Overriding is also possible via `packageOverrides` or using an overlay. 
**Note:** Flake support is experimental.
See examples above.

## Contribution Guidelines

*   Ensure packages build and set `meta.broken = true` if not.
*   Use standard meta attributes.
*   Keep repositories slim.
*   Reuse packages from Nixpkgs.

## Examples of Packages for NUR:

*   Packages for niche audiences.
*   Pre-releases.
*   Legacy package versions.
*   Auto-generated package sets.
*   Software with custom patches.
*   Experiments.

## Contact

Join the conversation on the matrix channel [#nur:nixos.org](https://matrix.to/#/#nur:nixos.org) or the NixOS Discourse.