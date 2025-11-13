"""
API Key Authentication System
- Auto-generated random keys
- 90-day expiration
- Persistent storage
"""

import json
import secrets
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

# Storage file for API keys
API_KEYS_FILE = Path(__file__).parent.parent.parent / "data" / "api_keys.json"
API_KEYS_FILE.parent.mkdir(parents=True, exist_ok=True)

# Default expiration: 90 days
DEFAULT_EXPIRATION_DAYS = 90


def generate_api_key() -> str:
    """
    Generate a secure random API key
    Format: 32 characters hex string
    """
    return secrets.token_hex(32)


def hash_api_key(api_key: str) -> str:
    """
    Hash API key for secure storage
    """
    return hashlib.sha256(api_key.encode()).hexdigest()


def load_api_keys() -> Dict[str, Dict[str, Any]]:
    """
    Load API keys from storage file
    """
    if not API_KEYS_FILE.exists():
        return {}
    
    try:
        with open(API_KEYS_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return {}


def save_api_keys(keys: Dict[str, Dict[str, Any]]):
    """
    Save API keys to storage file
    """
    try:
        with open(API_KEYS_FILE, 'w') as f:
            json.dump(keys, f, indent=2)
    except Exception as e:
        print(f"Error saving API keys: {e}")


def create_api_key(
    name: str,
    expiration_days: int = DEFAULT_EXPIRATION_DAYS,
    metadata: Optional[Dict] = None
) -> str:
    """
    Create a new API key
    
    Args:
        name: User/client name
        expiration_days: Days until expiration (default: 90)
        metadata: Additional metadata (optional)
    
    Returns:
        The generated API key (plain text, only shown once!)
    """
    # Generate new key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)
    
    # Calculate expiration
    created_at = datetime.now()
    expires_at = created_at + timedelta(days=expiration_days)
    
    # Load existing keys
    keys = load_api_keys()
    
    # Store key info (hashed)
    keys[key_hash] = {
        "name": name,
        "created_at": created_at.isoformat(),
        "expires_at": expires_at.isoformat(),
        "active": True,
        "metadata": metadata or {}
    }
    
    # Save to file
    save_api_keys(keys)
    
    print(f"✓ API key created for '{name}'")
    print(f"  Expires: {expires_at.strftime('%Y-%m-%d')}")
    print(f"  Key: {api_key}")
    print(f"  ⚠️  Save this key! It won't be shown again.")
    
    return api_key


def verify_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """
    Verify an API key
    
    Returns:
        Key info if valid, None if invalid/expired
    """
    key_hash = hash_api_key(api_key)
    keys = load_api_keys()
    
    # Check if key exists
    if key_hash not in keys:
        return None
    
    key_info = keys[key_hash]
    
    # Check if active
    if not key_info.get("active", False):
        return None
    
    # Check expiration
    expires_at = datetime.fromisoformat(key_info["expires_at"])
    if datetime.now() > expires_at:
        # Key expired
        return None
    
    return key_info


def list_api_keys() -> Dict[str, Dict[str, Any]]:
    """
    List all API keys (without showing actual keys)
    """
    keys = load_api_keys()
    
    # Add status to each key
    for key_hash, info in keys.items():
        expires_at = datetime.fromisoformat(info["expires_at"])
        days_left = (expires_at - datetime.now()).days
        
        info["days_until_expiration"] = days_left
        info["expired"] = days_left < 0
        info["key_hash"] = key_hash[:16] + "..."  # Show partial hash
    
    return keys


def revoke_api_key(key_hash: str):
    """
    Revoke (deactivate) an API key
    """
    keys = load_api_keys()
    
    if key_hash in keys:
        keys[key_hash]["active"] = False
        save_api_keys(keys)
        print(f"✓ API key revoked: {key_hash[:16]}...")
        return True
    
    print(f"✗ API key not found: {key_hash[:16]}...")
    return False


def cleanup_expired_keys():
    """
    Remove expired keys from storage
    """
    keys = load_api_keys()
    now = datetime.now()
    
    # Find expired keys
    expired = []
    for key_hash, info in keys.items():
        expires_at = datetime.fromisoformat(info["expires_at"])
        if now > expires_at:
            expired.append(key_hash)
    
    # Remove expired keys
    for key_hash in expired:
        del keys[key_hash]
    
    if expired:
        save_api_keys(keys)
        print(f"✓ Cleaned up {len(expired)} expired keys")
    
    return len(expired)


def renew_api_key(key_hash: str, days: int = DEFAULT_EXPIRATION_DAYS) -> bool:
    """
    Extend expiration of an existing key
    """
    keys = load_api_keys()
    
    if key_hash in keys:
        new_expiration = datetime.now() + timedelta(days=days)
        keys[key_hash]["expires_at"] = new_expiration.isoformat()
        save_api_keys(keys)
        print(f"✓ API key renewed until {new_expiration.strftime('%Y-%m-%d')}")
        return True
    
    return False


# =============================================================================
# CLI Management Tool
# =============================================================================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="API Key Management")
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Create key
    create_parser = subparsers.add_parser("create", help="Create new API key")
    create_parser.add_argument("name", help="User/client name")
    create_parser.add_argument("--days", type=int, default=90, help="Expiration days (default: 90)")
    
    # List keys
    list_parser = subparsers.add_parser("list", help="List all API keys")
    
    # Revoke key
    revoke_parser = subparsers.add_parser("revoke", help="Revoke API key")
    revoke_parser.add_argument("key_hash", help="Key hash to revoke")
    
    # Cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Remove expired keys")
    
    # Renew key
    renew_parser = subparsers.add_parser("renew", help="Renew API key")
    renew_parser.add_argument("key_hash", help="Key hash to renew")
    renew_parser.add_argument("--days", type=int, default=90, help="Extension days")
    
    args = parser.parse_args()
    
    if args.command == "create":
        create_api_key(args.name, args.days)
    
    elif args.command == "list":
        keys = list_api_keys()
        print("\n" + "="*80)
        print("API KEYS")
        print("="*80)
        for key_hash, info in keys.items():
            status = "✓ Active" if info["active"] and not info["expired"] else "✗ Inactive/Expired"
            print(f"\n{status}")
            print(f"  Name: {info['name']}")
            print(f"  Hash: {info['key_hash']}")
            print(f"  Created: {info['created_at'][:10]}")
            print(f"  Expires: {info['expires_at'][:10]}")
            print(f"  Days left: {info['days_until_expiration']}")
        print("="*80)
    
    elif args.command == "revoke":
        revoke_api_key(args.key_hash)
    
    elif args.command == "cleanup":
        cleanup_expired_keys()
    
    elif args.command == "renew":
        renew_api_key(args.key_hash, args.days)
    
    else:
        parser.print_help()

