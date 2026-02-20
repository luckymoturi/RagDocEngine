"""
Text-to-Speech Service for IntelliDoc
Converts text responses to audio using gTTS (Google Text-to-Speech)
"""

from gtts import gTTS
import os
import tempfile
import hashlib
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class TTSService:
    def __init__(self, cache_dir: str = "./tts_cache"):
        """Initialize TTS service with caching"""
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(f"TTS Service initialized with cache directory: {cache_dir}")

    def _get_cache_key(self, text: str, lang: str = "en") -> str:
        """Generate cache key from text and language"""
        content = f"{text}_{lang}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_path(self, cache_key: str) -> str:
        """Get full path for cached audio file"""
        return os.path.join(self.cache_dir, f"{cache_key}.mp3")

    def text_to_speech(
        self, 
        text: str, 
        lang: str = "en", 
        slow: bool = False,
        use_cache: bool = True
    ) -> str:
        """
        Convert text to speech and return file path
        
        Args:
            text: Text to convert to speech
            lang: Language code (default: 'en')
            slow: Whether to use slow speech rate
            use_cache: Whether to use cached audio if available
            
        Returns:
            Path to the generated audio file
        """
        try:
            # Check cache first
            if use_cache:
                cache_key = self._get_cache_key(text, lang)
                cache_path = self._get_cache_path(cache_key)
                
                if os.path.exists(cache_path):
                    logger.info(f"Using cached audio for text: {text[:50]}...")
                    return cache_path

            # Generate new audio
            logger.info(f"Generating audio for text: {text[:50]}...")
            tts = gTTS(text=text, lang=lang, slow=slow)
            
            # Save to cache
            if use_cache:
                cache_key = self._get_cache_key(text, lang)
                cache_path = self._get_cache_path(cache_key)
                tts.save(cache_path)
                logger.info(f"Audio saved to cache: {cache_path}")
                return cache_path
            else:
                # Save to temporary file
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                tts.save(temp_file.name)
                logger.info(f"Audio saved to temp file: {temp_file.name}")
                return temp_file.name
                
        except Exception as e:
            logger.error(f"Error generating speech: {str(e)}")
            raise Exception(f"TTS generation failed: {str(e)}")

    def clean_cache(self, max_files: int = 100):
        """Clean old cache files if cache size exceeds limit"""
        try:
            cache_files = [
                os.path.join(self.cache_dir, f) 
                for f in os.listdir(self.cache_dir) 
                if f.endswith('.mp3')
            ]
            
            if len(cache_files) > max_files:
                # Sort by modification time
                cache_files.sort(key=lambda x: os.path.getmtime(x))
                
                # Remove oldest files
                files_to_remove = cache_files[:len(cache_files) - max_files]
                for file_path in files_to_remove:
                    os.remove(file_path)
                    logger.info(f"Removed old cache file: {file_path}")
                    
                logger.info(f"Cache cleaned: removed {len(files_to_remove)} files")
                
        except Exception as e:
            logger.error(f"Error cleaning cache: {str(e)}")

    def get_cache_stats(self) -> dict:
        """Get statistics about the cache"""
        try:
            cache_files = [
                os.path.join(self.cache_dir, f) 
                for f in os.listdir(self.cache_dir) 
                if f.endswith('.mp3')
            ]
            
            total_size = sum(os.path.getsize(f) for f in cache_files)
            
            return {
                "total_files": len(cache_files),
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "cache_directory": self.cache_dir
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {str(e)}")
            return {"error": str(e)}
