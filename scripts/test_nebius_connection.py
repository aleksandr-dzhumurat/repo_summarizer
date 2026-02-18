"""Debug script to test Nebius TokenFactory API connection and available models."""

import asyncio
import json
import os

import httpx


async def test_nebius_connection():
    """Test connection to Nebius TokenFactory and check available models."""
    api_key = os.getenv("NEBIUS_API_KEY", "")
    
    if not api_key:
        print("ERROR: NEBIUS_API_KEY not set")
        return
    
    print("Testing Nebius TokenFactory API connection...")
    print(f"API Key: {api_key[:20]}..." if len(api_key) > 20 else f"API Key: {api_key}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    # Try a simple test call
    payload = {
        "model": "deepseek-ai/DeepSeek-V3.2",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }
    
    print(f"\n1. Testing with model: {payload['model']}")
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            response = await client.post(
                "https://api.tokenfactory.nebius.com/v1/chat/completions",
                json=payload,
                headers=headers,
            )
            print(f"   Status: {response.status_code}")
            if response.status_code != 200:
                print(f"   Response: {response.text}")
            else:
                print(f"   Success! Response: {response.json()}")
        except Exception as e:
            print(f"   Error: {e}")
    
    # Try alternative model names
    model_variants = [
        "DeepSeek-V3.2",
        "deepseek-v3.2",
        "DeepSeek-V3",
        "deepseek-v3",
    ]
    
    print("\n2. Testing alternative model names:")
    for model_name in model_variants:
        payload["model"] = model_name
        async with httpx.AsyncClient(timeout=30) as client:
            try:
                response = await client.post(
                    "https://api.tokenfactory.nebius.com/v1/chat/completions",
                    json=payload,
                    headers=headers,
                )
                status = response.status_code
                if status == 200:
                    print(f"   ✓ {model_name}: SUCCESS")
                    # Print actual response structure
                    data = response.json()
                    print(f"     Model used: {data.get('model')}")
                else:
                    error = json.loads(response.text).get("detail", response.text)
                    print(f"   ✗ {model_name}: {status} - {error}")
            except Exception as e:
                print(f"   ✗ {model_name}: Connection error - {e}")


if __name__ == "__main__":
    asyncio.run(test_nebius_connection())
