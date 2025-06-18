"""Factory for managing AutoProcessor instances with singleton pattern."""
import logging
from typing import Dict, Optional
from transformers import AutoProcessor

logger = logging.getLogger(__name__)

class ProcessorFactory:
    """Singleton factory for managing AutoProcessor instances."""
    
    _instance: Optional['ProcessorFactory'] = None
    _processors: Dict[str, AutoProcessor] = {}
    
    def __new__(cls) -> 'ProcessorFactory':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_processor(self, model_name: str) -> AutoProcessor:
        """Get or create an AutoProcessor for the given model.
        
        Args:
            model_name: The name of the model to load the processor for.
            
        Returns:
            The loaded AutoProcessor instance.
            
        Raises:
            Exception: If processor loading fails.
        """
        if model_name not in self._processors:
            logger.info(f"Loading processor for {model_name}")
            try:
                self._processors[model_name] = AutoProcessor.from_pretrained(model_name)
                logger.info(f"Successfully loaded processor for {model_name}")
            except Exception as e:
                logger.error(f"Failed to load processor for {model_name}: {e}")
                raise
        
        return self._processors[model_name]
    
    def clear_cache(self) -> None:
        """Clear all cached processors."""
        self._processors.clear()
        logger.info("Cleared all cached processors")
    
    def remove_processor(self, model_name: str) -> None:
        """Remove a specific processor from cache.
        
        Args:
            model_name: The model name whose processor should be removed.
        """
        if model_name in self._processors:
            del self._processors[model_name]
            logger.info(f"Removed processor for {model_name}")

# Global factory instance
processor_factory = ProcessorFactory()