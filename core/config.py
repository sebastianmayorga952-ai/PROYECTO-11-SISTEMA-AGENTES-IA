"""
core/config.py - Gestión de configuración del sistema.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Ruta por defecto del archivo de configuración
DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "settings.yaml"


class LLMConfig(BaseModel):
    """Configuración del proveedor LLM."""

    provider: str = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "codellama:13b"
    planning_model: str = "llama3:8b"
    review_model: str = "codellama:13b"
    temperature: float = 0.3
    max_tokens: int = 4096
    timeout: int = 120


class OpenCodeConfig(BaseModel):
    """Configuración de OpenCode CLI."""

    binary: str = "opencode"
    mode: str = "build"
    auto_approve: bool = False
    workspace: str = "./output"


class AgentsConfig(BaseModel):
    """Configuración de agentes."""

    max_concurrent: int = 3
    retry_attempts: int = 2
    communication_mode: str = "event_bus"


class LoggingConfig(BaseModel):
    """Configuración de logging."""

    level: str = "INFO"
    file: str = "logs/system.log"
    format: str = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"


class OutputConfig(BaseModel):
    """Configuración de salida."""

    base_dir: str = "./output"
    create_readme: bool = True
    create_tests: bool = True
    git_init: bool = True


class SystemConfig(BaseModel):
    """Configuración completa del sistema."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    opencode: OpenCodeConfig = Field(default_factory=OpenCodeConfig)
    agents: AgentsConfig = Field(default_factory=AgentsConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    Cargar configuración desde archivo YAML.

    Args:
        config_path: Ruta al archivo de configuración

    Returns:
        SystemConfig con la configuración cargada
    """
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if path.exists():
        logger.info(f"Cargando configuración desde: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
            if raw:
                return SystemConfig(**raw)

    logger.warning(f"Archivo no encontrado: {path}. Usando configuración por defecto.")
    return SystemConfig()


def setup_logging(config: LoggingConfig) -> None:
    """Configurar el sistema de logging."""
    log_file = Path(config.file)
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, config.level.upper(), logging.INFO),
        format=config.format,
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
