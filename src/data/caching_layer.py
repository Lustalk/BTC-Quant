"""
Caching Layer Module
Professional Redis-based caching for financial data.
"""

import redis
import json
import pickle
from typing import Any, Optional, Dict, Union
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class CacheManager:
    """Professional Redis-based cache manager for financial data."""
    
    def __init__(self, redis_client: Optional[redis.Redis] = None, 
                 default_ttl: int = 3600):
        """Initialize cache manager with Redis client."""
        self.redis_client = redis_client
        self.default_ttl = default_ttl
        
        if redis_client is None:
            logger.warning("No Redis client provided, using mock cache")
            self._mock_cache = {}
    
    def _get_key(self, prefix: str, identifier: str) -> str:
        """Generate cache key with prefix."""
        return f"{prefix}:{identifier}"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache with TTL."""
        try:
            if self.redis_client:
                # Serialize value
                if isinstance(value, (dict, list)):
                    serialized = json.dumps(value)
                else:
                    serialized = pickle.dumps(value)
                
                return self.redis_client.setex(
                    key, 
                    ttl or self.default_ttl, 
                    serialized
                )
            else:
                # Mock cache
                self._mock_cache[key] = {
                    'value': value,
                    'expires': datetime.now() + timedelta(seconds=ttl or self.default_ttl)
                }
                return True
        except Exception as e:
            logger.error(f"Cache set error: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        try:
            if self.redis_client:
                value = self.redis_client.get(key)
                if value is None:
                    return None
                
                # Try to deserialize
                try:
                    return json.loads(value)
                except:
                    return pickle.loads(value)
            else:
                # Mock cache
                if key in self._mock_cache:
                    cache_entry = self._mock_cache[key]
                    if datetime.now() < cache_entry['expires']:
                        return cache_entry['value']
                    else:
                        del self._mock_cache[key]
                return None
        except Exception as e:
            logger.error(f"Cache get error: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.delete(key))
            else:
                # Mock cache
                if key in self._mock_cache:
                    del self._mock_cache[key]
                return True
        except Exception as e:
            logger.error(f"Cache delete error: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        try:
            if self.redis_client:
                return bool(self.redis_client.exists(key))
            else:
                return key in self._mock_cache
        except Exception as e:
            logger.error(f"Cache exists error: {e}")
            return False
    
    def set_bitcoin_data(self, symbol: str, data: Dict, ttl: int = 3600) -> bool:
        """Cache Bitcoin price data."""
        key = self._get_key("bitcoin", symbol)
        return self.set(key, data, ttl)
    
    def get_bitcoin_data(self, symbol: str) -> Optional[Dict]:
        """Get cached Bitcoin price data."""
        key = self._get_key("bitcoin", symbol)
        return self.get(key)
    
    def clear_bitcoin_cache(self, symbol: str) -> bool:
        """Clear Bitcoin data cache."""
        key = self._get_key("bitcoin", symbol)
        return self.delete(key)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        try:
            if self.redis_client:
                info = self.redis_client.info()
                return {
                    'total_keys': info.get('db0', {}).get('keys', 0),
                    'memory_usage': info.get('used_memory_human', 'N/A'),
                    'hit_rate': info.get('keyspace_hits', 0) / max(info.get('keyspace_misses', 1), 1)
                }
            else:
                return {
                    'total_keys': len(self._mock_cache),
                    'memory_usage': 'N/A (mock)',
                    'hit_rate': 0.0
                }
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {}

def create_cache_manager(redis_url: Optional[str] = None) -> CacheManager:
    """Factory function to create cache manager."""
    if redis_url:
        try:
            redis_client = redis.from_url(redis_url)
            return CacheManager(redis_client)
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
    
    return CacheManager() 