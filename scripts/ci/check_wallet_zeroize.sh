#!/usr/bin/env bash
# CP-B-005 tripwire: decrypted secp256k1 key material in the keystore-decrypt
# path must not be left in un-zeroized heap/stack. Both wallet copies must wrap
# the PBKDF2-derived key, the decrypted plaintext, and the hex round-trip string
# in `zeroize::Zeroizing` so they are wiped on drop.
#
# RED (pinned code): no `zeroize` import; `plaintext`/`hex_key` are bare.
# GREEN (fixed):      both files import Zeroizing and wrap the intermediates.
set -euo pipefail
cd "$(dirname "$0")/../.."

fail=0
for f in training-worker/src/wallet.rs pool-coordinator/src/wallet.rs; do
  if ! grep -q 'use zeroize::Zeroizing;' "$f"; then
    echo "FAIL: $f does not import zeroize::Zeroizing"
    fail=1
  fi
  # The raw, un-wrapped decrypt intermediates must be gone.
  if grep -q 'let mut plaintext = ciphertext.clone();' "$f"; then
    echo "FAIL: $f leaves decrypted plaintext un-zeroized (raw Vec clone)"
    fail=1
  fi
  if grep -qE 'let hex_key = hex::encode\(&plaintext\);' "$f"; then
    echo "FAIL: $f round-trips the key through an un-zeroized hex String"
    fail=1
  fi
  # And the wrapped forms must be present.
  if ! grep -q 'Zeroizing::new(hex::encode' "$f"; then
    echo "FAIL: $f does not wrap the hex key in Zeroizing"
    fail=1
  fi
done

if [ "$fail" -ne 0 ]; then
  echo "CP-B-005 tripwire: RED (key material not zeroized)"
  exit 1
fi
echo "CP-B-005 tripwire: GREEN (key material wrapped in Zeroizing in both wallets)"
