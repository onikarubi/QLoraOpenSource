from typing import Dict, Optional


class ModelRegistry:
    """A registry to manage model configurations."""

    def __init__(self) -> None:
        self.models: Dict[str, Dict] = {
            "gemma": {
                "default": "google/gemma-2-2b-it",
            },
            "llama": {
                "default": "meta-llama/Meta-Llama-3-8B-Instruct",
            },
            "elyza": {
                "default": "elyza/Llama-3-ELYZA-JP-8B",
                "llama2": {
                    "default": "elyza/ELYZA-japanese-Llama-2-7b",
                    "instruct": "elyza/ELYZA-japanese-Llama-2-7b-instruct",
                },
                "llama3": {
                    "default": "elyza/Llama-3-ELYZA-JP-8B",
                },
            },
        }

    def get_model(
        self,
        family: str,
        version: Optional[str] = None,
        variant: Optional[str] = "default",
    ) -> Optional[str]:
        """Retrieve a model configuration from the registry.

        Args:
            family (str): The family of the model (e.g., 'gemma', 'llama', 'elyza').
            version (Optional[str]): The specific version or subcategory (e.g., 'llama2'). Defaults to None.
            variant (Optional[str]): The specific model variant (e.g., 'default', 'instruct'). Defaults to "default".

        Returns:
            Optional[str]: The model identifier or None if not found.
        """
        try:
            if version:
                return self.models[family][version][variant]
            return self.models[family][variant]
        except KeyError:
            return None

    def add_model(
        self, family: str, version: Optional[str], variant: str, model_id: str
    ) -> None:
        """Add a new model configuration to the registry.

        Args:
            family (str): The family of the model (e.g., 'gemma', 'llama', 'elyza').
            version (Optional[str]): The specific version or subcategory (e.g., 'llama2'). Can be None.
            variant (str): The specific model variant (e.g., 'default', 'instruct').
            model_id (str): The model identifier to be added.
        """
        if family not in self.models:
            self.models[family] = {}

        if version:
            if version not in self.models[family]:
                self.models[family][version] = {}
            self.models[family][version][variant] = model_id
        else:
            self.models[family][variant] = model_id

    def remove_model(
        self, family: str, version: Optional[str] = None, variant: str = "default"
    ) -> bool:
        """Remove a model configuration from the registry.

        Args:
            family (str): The family of the model (e.g., 'gemma', 'llama', 'elyza').
            version (Optional[str]): The specific version or subcategory (e.g., 'llama2'). Defaults to None.
            variant (str): The specific model variant (e.g., 'default', 'instruct'). Defaults to "default".

        Returns:
            bool: True if the model was removed, False if not found.
        """
        try:
            if version:
                del self.models[family][version][variant]
                if not self.models[family][version]:  # Remove empty dictionaries
                    del self.models[family][version]
            else:
                del self.models[family][variant]
            return True
        except KeyError:
            return False

    def list_models(self) -> Dict[str, Dict]:
        """List all models in the registry.

        Returns:
            Dict[str, Dict]: A dictionary containing all model configurations.
        """
        return self.models
