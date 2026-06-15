#!/usr/bin/env python3
"""
Generate VAPID key pair for Web Push notifications.
Run once, then add the printed values to Railway → Variables.

  python generate_vapid_keys.py
"""
import base64
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

key = ec.generate_private_key(ec.SECP256R1(), default_backend())

# Private key — PEM format for pywebpush (paste multi-line value into Railway)
private_pem = key.private_bytes(
    encoding=serialization.Encoding.PEM,
    format=serialization.PrivateFormat.TraditionalOpenSSL,
    encryption_algorithm=serialization.NoEncryption(),
).decode().strip()

# Public key — uncompressed P-256 point, base64url-encoded (for browser PushManager)
pub_raw = key.public_key().public_bytes(
    encoding=serialization.Encoding.X962,
    format=serialization.PublicFormat.UncompressedPoint,
)
public_b64 = base64.urlsafe_b64encode(pub_raw).rstrip(b"=").decode()

print("=" * 60)
print("Add these variables to Railway → your service → Variables")
print("=" * 60)
print()
print(f"VAPID_PUBLIC_KEY={public_b64}")
print()
print("VAPID_PRIVATE_KEY=")
print(private_pem)
print()
print("VAPID_EMAIL=mailto:your@email.com")
print()
print("=" * 60)
print("NOTE: Paste the full PEM block (including BEGIN/END lines)")
print("      as a single multi-line variable value in Railway.")
print("=" * 60)
