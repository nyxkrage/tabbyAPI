import aiofiles
import json
import pathlib
from os import path
from loguru import logger
from pydantic import BaseModel
from typing import Dict, List, Optional, Set, Union, Tuple

from huggingface_hub.constants import HF_HUB_CACHE
from huggingface_hub.file_download import repo_folder_name

class GenerationConfig(BaseModel):
    """
    An abridged version of HuggingFace's GenerationConfig.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        generation_config_path = model_directory / "generation_config.json"
        async with aiofiles.open(
            generation_config_path, "r", encoding="utf8"
        ) as generation_config_json:
            contents = await generation_config_json.read()
            generation_config_dict = json.loads(contents)
            return cls.model_validate(generation_config_dict)

    def eos_tokens(self):
        """Wrapper method to fetch EOS tokens."""

        if isinstance(self.eos_token_id, list):
            return self.eos_token_id
        elif isinstance(self.eos_token_id, int):
            return [self.eos_token_id]
        else:
            return []


class HuggingFaceConfig(BaseModel):
    """
    DEPRECATED: Currently a stub and doesn't do anything.

    An abridged version of HuggingFace's model config.
    Will be expanded as needed.
    """

    eos_token_id: Optional[Union[int, List[int]]] = None
    quantization_config: Optional[Dict] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a generation config file."""

        hf_config_path = model_directory / "config.json"
        async with aiofiles.open(
            hf_config_path, "r", encoding="utf8"
        ) as hf_config_json:
            contents = await hf_config_json.read()
            hf_config_dict = json.loads(contents)
            return cls.model_validate(hf_config_dict)

    def quant_method(self):
        """Wrapper method to fetch quant type"""

        if isinstance(self.quantization_config, Dict):
            return self.quantization_config.get("quant_method")
        else:
            return None

    def eos_tokens(self):
        """Wrapper method to fetch EOS tokens."""

        if isinstance(self.eos_token_id, list):
            return self.eos_token_id
        elif isinstance(self.eos_token_id, int):
            return [self.eos_token_id]
        else:
            return []


class TokenizerConfig(BaseModel):
    """
    An abridged version of HuggingFace's tokenizer config.
    """

    add_bos_token: Optional[bool] = True

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a tokenizer config file."""

        tokenizer_config_path = model_directory / "tokenizer_config.json"
        async with aiofiles.open(
            tokenizer_config_path, "r", encoding="utf8"
        ) as tokenizer_config_json:
            contents = await tokenizer_config_json.read()
            tokenizer_config_dict = json.loads(contents)
            return cls.model_validate(tokenizer_config_dict)


class HFModel:
    """
    Unified container for HuggingFace model configuration files.
    These are abridged for hyper-specific model parameters not covered
    by most backends.

    Includes:
      - config.json
      - generation_config.json
      - tokenizer_config.json
    """

    hf_config: HuggingFaceConfig
    repo_id: Optional[str] = None
    revision: Optional[str] = None
    tokenizer_config: Optional[TokenizerConfig] = None
    generation_config: Optional[GenerationConfig] = None

    @classmethod
    async def from_directory(cls, model_directory: pathlib.Path):
        """Create an instance from a model directory"""

        self = cls()

        # A model must have an HF config
        try:
            self.hf_config = await HuggingFaceConfig.from_directory(model_directory)
        except Exception as exc:
            raise ValueError(
                f"Failed to load config.json from {model_directory}"
            ) from exc

        try:
            self.generation_config = await GenerationConfig.from_directory(
                model_directory
            )
        except Exception:
            logger.warning(
                "Generation config file not found in model directory, skipping."
            )

        try:
            self.tokenizer_config = await TokenizerConfig.from_directory(
                model_directory
            )
        except Exception:
            logger.warning(
                "Tokenizer config file not found in model directory, skipping."
            )

        try:
            hf_model_repo_info = get_hf_cache_model(model_directory)
            if hf_model_repo_info is None:
                logger.warning(
                    "Failed to get repo_id and revision from model directory, skipping."
                )
            else:
                self.repo_id, self.revision = hf_model_repo_info
        except Exception:
            logger.warning(
                "Failed to get repo_id and revision from model directory, skipping."
            )
            pass

        return self

    def quant_method(self):
        """Wrapper for quantization method"""

        return self.hf_config.quant_method()

    def eos_tokens(self):
        """Combines and returns EOS tokens from various configs"""

        eos_ids: Set[int] = set()

        eos_ids.update(self.hf_config.eos_tokens())

        if self.generation_config:
            eos_ids.update(self.generation_config.eos_tokens())

        # Convert back to a list
        return list(eos_ids)

    def add_bos_token(self):
        """Wrapper for tokenizer config"""

        if self.tokenizer_config:
            return self.tokenizer_config.add_bos_token

        # Expected default
        return True

def get_hf_cache_path(repo_id: str, revision: str) -> Optional[pathlib.Path]:
    """
    Get a HuggingFace model from the cache.
    """
    cache_dir = pathlib.Path(HF_HUB_CACHE).expanduser().resolve()
    repo_dir = cache_dir / repo_folder_name(repo_id=repo_id, repo_type="model")
    with open(repo_dir / "refs" / revision, "r") as f:
        snapshot = f.read()
    snapshot_dir = repo_dir / "snapshots" / snapshot
    model_path = pathlib.Path(snapshot_dir)
    if not model_path.exists():
        logger.warning(
            "Model loading does not support downloading from huggingface. "
            "Please download the model manually using huggingface-cli. "
        )
        return None
    return model_path

def get_hf_cache_model(model_path: pathlib.Path) -> Optional[Tuple[str,str]]:
    """
    Get a HuggingFace model from the cache.
    """
    hf_model_dir_name = pathlib.Path(path.relpath(model_path, HF_HUB_CACHE)).parts[0]
    hf_model_dir = pathlib.Path(HF_HUB_CACHE) / hf_model_dir_name
    snapshot = model_path.name
    repo_id = "/".join(hf_model_dir.name.split("--")[1:])
    revision = None
    for ref in hf_model_dir.glob("refs/*"):
        with open(ref, "r") as f:
            ref_snapshot = f.read()
            if ref_snapshot == snapshot:
                revision = ref.name
                break

    if revision is None:
        return None
    return repo_id, revision
