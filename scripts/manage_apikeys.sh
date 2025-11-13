#!/bin/bash

# API Key Management Script
# Wrapper for easy API key management

cd "$(dirname "$0")/.." || exit 1

echo "=========================================="
echo "API Key Management"
echo "=========================================="
echo ""

case "$1" in
    create)
        if [ -z "$2" ]; then
            echo "Usage: $0 create <name> [days]"
            echo "Example: $0 create \"Client ABC\" 90"
            exit 1
        fi
        
        NAME="$2"
        DAYS="${3:-90}"
        
        echo "Creating API key for: $NAME"
        echo "Expiration: $DAYS days"
        echo ""
        python -m src.api.auth create "$NAME" --days "$DAYS"
        ;;
    
    list)
        python -m src.api.auth list
        ;;
    
    revoke)
        if [ -z "$2" ]; then
            echo "Usage: $0 revoke <key_hash>"
            echo "Get key_hash from 'list' command"
            exit 1
        fi
        
        python -m src.api.auth revoke "$2"
        ;;
    
    cleanup)
        echo "Cleaning up expired keys..."
        python -m src.api.auth cleanup
        ;;
    
    renew)
        if [ -z "$2" ]; then
            echo "Usage: $0 renew <key_hash> [days]"
            echo "Example: $0 renew abc123... 90"
            exit 1
        fi
        
        DAYS="${3:-90}"
        python -m src.api.auth renew "$2" --days "$DAYS"
        ;;
    
    *)
        echo "Usage: $0 {create|list|revoke|cleanup|renew}"
        echo ""
        echo "Commands:"
        echo "  create <name> [days]   - Create new API key (default: 90 days)"
        echo "  list                   - List all API keys"
        echo "  revoke <hash>          - Revoke an API key"
        echo "  cleanup                - Remove expired keys"
        echo "  renew <hash> [days]    - Extend key expiration"
        echo ""
        echo "Examples:"
        echo "  $0 create \"Client ABC\" 90"
        echo "  $0 list"
        echo "  $0 revoke abc123..."
        echo "  $0 renew abc123... 180"
        exit 1
        ;;
esac

