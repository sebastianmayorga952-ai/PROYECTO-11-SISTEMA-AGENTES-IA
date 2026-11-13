"""
core/event_bus.py - Sistema de eventos para comunicación entre agentes.

El Event Bus es el corazón de la comunicación del sistema. Los agentes
publican y se suscriben a eventos tipados para coordinar su trabajo
de forma desacoplada.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Tipos de eventos del sistema."""

    # Ciclo de vida del proyecto
    PROJECT_CREATED = "project.created"
    PROJECT_COMPLETED = "project.completed"
    PROJECT_FAILED = "project.failed"

    # Planificación
    PLAN_REQUESTED = "plan.requested"
    PLAN_GENERATED = "plan.generated"
    PLAN_APPROVED = "plan.approved"

    # Construcción
    BUILD_STARTED = "build.started"
    BUILD_FILE_CREATED = "build.file_created"
    BUILD_COMPLETED = "build.completed"
    BUILD_FAILED = "build.failed"

    # Revisión
    REVIEW_REQUESTED = "review.requested"
    REVIEW_COMPLETED = "review.completed"
    REVIEW_ISSUES_FOUND = "review.issues_found"

    # Skills
    SKILL_STARTED = "skill.started"
    SKILL_COMPLETED = "skill.completed"
    SKILL_FAILED = "skill.failed"

    # OpenCode
    OPENCODE_COMMAND_SENT = "opencode.command_sent"
    OPENCODE_RESPONSE_RECEIVED = "opencode.response_received"
    OPENCODE_ERROR = "opencode.error"

    # Sistema
    AGENT_READY = "agent.ready"
    AGENT_BUSY = "agent.busy"
    AGENT_ERROR = "agent.error"
    SYSTEM_SHUTDOWN = "system.shutdown"


@dataclass
class Event:
    """Evento del sistema con metadatos."""

    type: EventType
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None

    def __repr__(self) -> str:
        return f"Event({self.type.value}, source={self.source}, ts={self.timestamp.isoformat()})"


# Tipo para handlers de eventos
EventHandler = Callable[[Event], Coroutine[Any, Any, None]]


class EventBus:
    """
    Bus de eventos asíncrono para comunicación entre agentes.

    Soporta:
      - Publicación/suscripción tipada
      - Handlers async
      - Historial de eventos
      - Filtros por source
    """

    def __init__(self, max_history: int = 1000):
        self._subscribers: Dict[EventType, List[EventHandler]] = {}
        self._global_subscribers: List[EventHandler] = []
        self._history: List[Event] = []
        self._max_history = max_history
        self._lock = asyncio.Lock()
        logger.info("EventBus inicializado")

    def subscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Suscribirse a un tipo de evento."""
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.debug(f"Handler suscrito a {event_type.value}")

    def subscribe_all(self, handler: EventHandler) -> None:
        """Suscribirse a todos los eventos."""
        self._global_subscribers.append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler) -> None:
        """Desuscribirse de un tipo de evento."""
        if event_type in self._subscribers:
            self._subscribers[event_type] = [
                h for h in self._subscribers[event_type] if h != handler
            ]

    async def publish(self, event: Event) -> None:
        """Publicar un evento a todos los suscriptores."""
        async with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        logger.info(f"Evento publicado: {event}")

        # Notificar suscriptores específicos
        handlers = self._subscribers.get(event.type, [])
        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error en handler de {event.type.value}: {e}")

        # Notificar suscriptores globales
        for handler in self._global_subscribers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Error en handler global: {e}")

    async def emit(
        self,
        event_type: EventType,
        data: Dict[str, Any] = None,
        source: str = "system",
        correlation_id: str = None,
    ) -> None:
        """Atajo para crear y publicar un evento."""
        event = Event(
            type=event_type,
            data=data or {},
            source=source,
            correlation_id=correlation_id,
        )
        await self.publish(event)

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
        """Obtener historial de eventos con filtros opcionales."""
        events = self._history
        if event_type:
            events = [e for e in events if e.type == event_type]
        if source:
            events = [e for e in events if e.source == source]
        return events[-limit:]

    def clear_history(self) -> None:
        """Limpiar historial de eventos."""
        self._history.clear()
