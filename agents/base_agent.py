"""
agents/base_agent.py - Clase base abstracta para todos los agentes.

Define la interfaz común, el ciclo de vida y la integración con
el Event Bus para comunicación inter-agentes.
"""

import asyncio
import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from core.event_bus import Event, EventBus, EventType
from core.llm_client import OllamaClient
from core.opencode_bridge import OpenCodeBridge

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Estados posibles de un agente."""
    IDLE = "idle"
    WORKING = "working"
    WAITING = "waiting"
    ERROR = "error"
    COMPLETED = "completed"


@dataclass
class AgentContext:
    """Contexto compartido entre agentes para un proyecto."""

    project_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    project_name: str = ""
    description: str = ""
    language: str = "python"
    project_dir: str = ""
    plan: Dict[str, Any] = field(default_factory=dict)
    files_created: List[str] = field(default_factory=list)
    review_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_id": self.project_id,
            "project_name": self.project_name,
            "description": self.description,
            "language": self.language,
            "project_dir": self.project_dir,
            "plan": self.plan,
            "files_created": self.files_created,
            "review_results": self.review_results,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Clase base abstracta para agentes del sistema.

    Cada agente:
      - Tiene un nombre y rol
      - Se comunica a través del EventBus
      - Puede usar OllamaClient para generar contenido
      - Puede usar OpenCodeBridge para ejecutar en el CLI
      - Mantiene un estado y un historial de acciones
    """

    def __init__(
        self,
        name: str,
        role: str,
        event_bus: EventBus,
        llm_client: OllamaClient,
        opencode: OpenCodeBridge,
    ):
        self.name = name
        self.role = role
        self.event_bus = event_bus
        self.llm = llm_client
        self.opencode = opencode
        self.state = AgentState.IDLE
        self.action_history: List[Dict[str, Any]] = []
        self._setup_subscriptions()
        logger.info(f"Agente '{self.name}' ({self.role}) inicializado")

    def _setup_subscriptions(self) -> None:
        """Configurar suscripciones a eventos. Override en subclases."""
        pass

    async def _log_action(self, action: str, details: Dict[str, Any] = None) -> None:
        """Registrar una acción en el historial."""
        entry = {
            "agent": self.name,
            "action": action,
            "details": details or {},
            "timestamp": datetime.now().isoformat(),
            "state": self.state.value,
        }
        self.action_history.append(entry)
        logger.info(f"[{self.name}] {action}")

    async def _set_state(self, state: AgentState) -> None:
        """Cambiar el estado del agente y notificar."""
        old_state = self.state
        self.state = state
        logger.debug(f"[{self.name}] Estado: {old_state.value} -> {state.value}")

        if state == AgentState.WORKING:
            await self.event_bus.emit(
                EventType.AGENT_BUSY, {"agent": self.name}, source=self.name
            )
        elif state == AgentState.ERROR:
            await self.event_bus.emit(
                EventType.AGENT_ERROR, {"agent": self.name}, source=self.name
            )

    @abstractmethod
    async def execute(self, context: AgentContext) -> AgentContext:
        """
        Ejecutar la tarea principal del agente.

        Args:
            context: Contexto del proyecto actual

        Returns:
            Contexto actualizado con los resultados
        """
        pass

    async def run(self, context: AgentContext) -> AgentContext:
        """
        Ejecutar el agente con manejo de errores y logging.

        Args:
            context: Contexto del proyecto

        Returns:
            Contexto actualizado
        """
        await self._set_state(AgentState.WORKING)
        await self._log_action("started", {"project_id": context.project_id})

        try:
            result = await self.execute(context)
            await self._set_state(AgentState.COMPLETED)
            await self._log_action("completed", {"project_id": context.project_id})
            return result

        except Exception as e:
            await self._set_state(AgentState.ERROR)
            await self._log_action("error", {"error": str(e)})
            logger.error(f"[{self.name}] Error: {e}", exc_info=True)
            raise

    def get_status(self) -> Dict[str, Any]:
        """Obtener estado actual del agente."""
        return {
            "name": self.name,
            "role": self.role,
            "state": self.state.value,
            "actions_count": len(self.action_history),
            "last_action": self.action_history[-1] if self.action_history else None,
        }
