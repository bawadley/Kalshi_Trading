"""Offline signature tests for Kalshi WebSocket authentication (no live credentials)."""
import base64
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
from cryptography.hazmat.primitives import hashes,serialization
from cryptography.hazmat.primitives.asymmetric import rsa,ed25519,padding
from v7_ws_recorder import headers

class AuthV7Tests(unittest.TestCase):
    def check_key(self,key):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"key.pem"
            p.write_bytes(key.private_bytes(serialization.Encoding.PEM,
                     serialization.PrivateFormat.PKCS8,serialization.NoEncryption()))
            with patch.dict("os.environ",{"KALSHI_API_KEY_ID":"test_key",
                         "KALSHI_PRIVATE_KEY_PATH":str(p)}), patch(
                         "v7_ws_recorder.time.time",return_value=1700000000.123):
                h=headers()
            self.assertEqual(h["KALSHI-ACCESS-KEY"],"test_key")
            self.assertEqual(h["KALSHI-ACCESS-TIMESTAMP"],"1700000000123")
            message=b"1700000000123GET/trade-api/ws/v2"
            sig=base64.b64decode(h["KALSHI-ACCESS-SIGNATURE"],validate=True)
            if isinstance(key,ed25519.Ed25519PrivateKey):
                key.public_key().verify(sig,message)
            else:
                key.public_key().verify(sig,message,padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.DIGEST_LENGTH),hashes.SHA256())

    def test_ed25519(self):
        self.check_key(ed25519.Ed25519PrivateKey.generate())

    def test_rsa(self):
        self.check_key(rsa.generate_private_key(public_exponent=65537,key_size=2048))

if __name__=="__main__": unittest.main()
