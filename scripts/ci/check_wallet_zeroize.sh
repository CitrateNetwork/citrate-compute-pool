#!/usr/bin/env bash
# CP-B-005 tripwire: decrypted secp256k1 key material in the keystore-decrypt
# path must not be left in un-zeroized heap/stack. Both wallet copies must wrap
# the PBKDF2-derived key, the decrypted plaintext, and the hex round-trip string
# in `zeroize::Zeroizing` so they are wiped on drop.
#
# RED (pinned code): no `zeroize` import; `plaintext`/`hex_key` are bare.
# GREEN (fixed):      both files import Zeroizing and wrap the intermediates.
set -euo pipefail
# Optional $1: the tree to check (default: this repo). Used by the negative
# control in pool-coordinator/tests/wallet_zeroize_tripwire.rs.
cd "${1:-$(dirname "$0")/../..}"

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
  # Every other copy of the key on the decrypt / from_hex path is wrapped too:
  # the PBKDF2-derived key, the decrypted plaintext, and the decoded key bytes.
  for wrapped in \
    'let mut derived = Zeroizing::new([0u8; 32]);' \
    'let mut plaintext = Zeroizing::new(ciphertext.clone());' \
    'Zeroizing::new(hex::decode(trimmed)' \
    'let mut sk_bytes = Zeroizing::new([0u8; 32]);'; do
    if ! grep -qF "$wrapped" "$f"; then
      echo "FAIL: $f is missing the wrapped form: $wrapped"
      fail=1
    fi
  done
done

if [ "$fail" -ne 0 ]; then
  echo "CP-B-005 tripwire: RED (key material not zeroized)"
  exit 1
fi
echo "CP-B-005 tripwire: GREEN (key material wrapped in Zeroizing in both wallets)"
