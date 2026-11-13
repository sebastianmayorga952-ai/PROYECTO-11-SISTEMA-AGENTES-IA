"""
agents/planner_agent.py - Agente Planificador.

AGENTE 1: El Planificador
=========================
Responsable de analizar los requisitos del usuario, descomponer
el proyecto en tareas, definir la arquitectura y crear un plan
de ejecución detallado.
"""

import json
import logging
from typing import Any, Dict, List

from agents.base_agent import AgentContext, BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)

# Prompt de sistema para el agente planificador
PLANNER_SYSTEM_PROMPT = """Eres un arquitecto de software experto y planificador de proyectos.
Tu tarea es analizar los requisitos del usuario y crear un plan de desarrollo detallado.

DEBES responder SIEMPRE en formato JSON válido con la siguiente estructura:
{
    "project_name": "nombre-del-proyecto",
    "summary": "Resumen del proyecto en 2-3 oraciones",
    "language": "lenguaje principal (python, javascript, etc.)",
    "architecture": {
        "pattern": "patrón de arquitectura (MVC, microservices, etc.)",
        "description": "descripción de la arquitectura"
    },
    "dependencies": ["lista", "de", "dependencias"],
    "files": [
        {
            "path": "ruta/al/archivo.py",
            "description": "Qué debe contener este archivo",
            "priority": 1
        }
    ],
    "tasks": [
        {
            "id": 1,
            "title": "Título de la tarea",
            "description": "Descripción detallada",
            "files": ["archivos involucrados"],
            "depends_on": []
        }
    ],
    "testing_strategy": "Estrategia de testing",
    "estimated_complexity": "low | medium | high"
}

REGLAS:
- Sé específico en las rutas de archivos
- Ordena las tareas por dependencias
- Incluye archivos de configuración, tests y documentación
- Estima la complejidad realísticamente
"""


class PlannerAgent(BaseAgent):
    """
    Agente Planificador - Analiza requisitos y genera planes de desarrollo.

    Responsabilidades:
      - Analizar la descripción del proyecto
      - Definir la arquitectura
      - Crear lista de archivos necesarios
      - Definir orden de tareas
      - Estimar complejidad
    """

    def __init__(self, event_bus, llm_client, opencode):
        super().__init__(
            name="planner",
            role="Arquitecto y Planificador",
            event_bus=event_bus,
            llm_client=llm_client,
            opencode=opencode,
        )

    def _setup_subscriptions(self) -> None:
        """Suscribirse a eventos relevantes."""
        self.event_bus.subscribe(EventType.PLAN_REQUESTED, self._on_plan_requested)
        self.event_bus.subscribe(
            EventType.REVIEW_ISSUES_FOUND, self._on_issues_found
        )

    async def _on_plan_requested(self, event: Event) -> None:
        """Handler cuando se solicita un plan."""
        logger.info(f"[{self.name}] Plan solicitado: {event.data}")

    async def _on_issues_found(self, event: Event) -> None:
        """Handler cuando se encuentran issues en revisión."""
        logger.info(
            f"[{self.name}] Issues encontrados, puede requerirse re-planificación"
        )

    async def execute(self, context: AgentContext) -> AgentContext:
        """
        Generar el plan de desarrollo.

        1. Analiza la descripción del proyecto
        2. Genera un plan estructurado con Ollama
        3. Opcionalmente usa OpenCode en modo plan para análisis adicional
        4. Publica el plan generado
        """
        await self._log_action("planning", {"description": context.description})

        # Publicar que se inició la planificación
        await self.event_bus.emit(
            EventType.PLAN_REQUESTED,
            {"description": context.description},
            source=self.name,
            correlation_id=context.project_id,
        )

        # Generar plan con Ollama
        prompt = self._build_planning_prompt(context)
        raw_plan = await self.llm.generate(
            prompt=prompt,
            system=PLANNER_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=4096,
        )

        # Parsear el plan JSON
        plan = self._parse_plan(raw_plan)

        if not plan:
            logger.warning("Plan no parseado correctamente, intentando con chat...")
            plan = await self._retry_with_chat(context)

        # Actualizar contexto
        context.plan = plan
        context.project_name = plan.get("project_name", context.project_name or "project")
        context.language = plan.get("language", context.language)

        # Publicar plan generado
        await self.event_bus.emit(
            EventType.PLAN_GENERATED,
            {"plan": plan, "project_id": context.project_id},
            source=self.name,
            correlation_id=context.project_id,
        )

        await self._log_action(
            "plan_generated",
            {
                "files_count": len(plan.get("files", [])),
                "tasks_count": len(plan.get("tasks", [])),
                "complexity": plan.get("estimated_complexity", "unknown"),
            },
        )

        return context

    def _build_planning_prompt(self, context: AgentContext) -> str:
        """Construir el prompt de planificación."""
        prompt = f"""Analiza los siguientes requisitos y genera un plan de desarrollo completo:

PROYECTO: {context.description}

LENGUAJE PREFERIDO: {context.language}

Requisitos adicionales:
- El código debe ser modular y bien organizado
- Incluir manejo de errores
- Incluir documentación (README, docstrings)
- Incluir tests unitarios básicos
- Seguir convenciones y mejores prácticas del lenguaje

Genera el plan en formato JSON como se especificó."""

        if context.metadata:
            prompt += f"\n\nMetadatos adicionales: {json.dumps(context.metadata)}"

        return prompt

    def _parse_plan(self, raw: str) -> Dict[str, Any]:
        """Parsear el plan desde la respuesta del LLM."""
        # Intentar extraer JSON del texto
        try:
            # Buscar bloque JSON en la respuesta
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                json_str = raw[start:end]
                return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"Error parseando JSON del plan: {e}")

        # Si falla, crear plan mínimo
        logger.warning("Usando plan mínimo por defecto")
        return self._default_plan()

    async def _retry_with_chat(self, context: AgentContext) -> Dict[str, Any]:
        """Reintentar generación de plan usando chat."""
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Genera un plan de desarrollo para: {context.description}\n"
                    f"Lenguaje: {context.language}\n"
                    f"Responde SOLO con el JSON, sin texto adicional."
                ),
            },
        ]

        raw = await self.llm.chat(messages, temperature=0.2)
        plan = self._parse_plan(raw)
        return plan if plan else self._default_plan()

    def _default_plan(self) -> Dict[str, Any]:
        """Plan mínimo por defecto."""
        return {
            "project_name": "project",
            "summary": "Proyecto generado automáticamente",
            "language": "python",
            "architecture": {"pattern": "modular", "description": "Arquitectura modular"},
            "dependencies": [],
            "files": [
                {"path": "main.py", "description": "Punto de entrada principal", "priority": 1},
                {"path": "README.md", "description": "Documentación del proyecto", "priority": 2},
            ],
            "tasks": [
                {
                    "id": 1,
                    "title": "Crear estructura base",
                    "description": "Crear archivos principales del proyecto",
                    "files": ["main.py"],
                    "depends_on": [],
                }
            ],
            "testing_strategy": "Unit tests con pytest",
            "estimated_complexity": "medium",
        }
