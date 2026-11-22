#!/usr/bin/env bash
# Builds the npm package `@sunsided/backshift` into bindings/wasm/pkg.
#
#     bindings/wasm/build.sh            # release build, ready for `npm pack`
#     bindings/wasm/build.sh --dev      # debug build, for local testing
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile=(--release)
if [[ "${1:-}" == "--dev" ]]; then profile=(--dev); fi

# `--target web`: an ES module that browsers load directly and bundlers accept;
# Node loads it with `initSync({ module: bytes })`.
wasm-pack build "$here" "${profile[@]}" --target web --out-dir pkg --out-name backshift

cd "$here/pkg"
# wasm-pack names the package after the crate; publish it under the short scoped name.
npm pkg set name=@sunsided/backshift
npm pkg set repository.type=git repository.url=git+https://github.com/sunsided/arima-rs.git
npm pkg set repository.directory=bindings/wasm
npm pkg set homepage=https://sunsided.github.io/arima-rs/
npm pkg set publishConfig.access=public
cp "$here/../../LICENSE-MIT" "$here/../../LICENSE-APACHE" .
npm pkg set 'files[]=LICENSE-MIT' 'files[]=LICENSE-APACHE'
# wasm-pack's own .gitignore would hide everything from `npm pack` in some setups.
rm -f .gitignore
