"""
skills/requirements_analysis.py - Skill de Análisis de Requisitos.

SKILL 1: Análisis de Requisitos
================================
Analiza la descripción en lenguaje natural del usuario y la transforma
en requisitos técnicos estructurados, identificando funcionalidades,
restricciones, y dependencias del proyecto.
"""

import json
import logging
from typing import Any, Dict, List, Optional

from core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

ANALYSIS_SYSTEM_PROMPT = """Eres un analista de requisitos de software experto.
Tu tarea es transformar descripciones en lenguaje natural en requisitos técnicos
estructurados y accionables.

Responde SIEMPRE en JSON con esta estructura:
{
    "project_type": "cli | web | api | desktop | library | script",
    "functional_requirements": [
        {
            "id": "FR-001",
            "title": "Título del requisito",
            "description": "Descripción detallada",
            "priority": "must | should | could",
            "acceptance_criteria": ["criterio 1", "criterio 2"]
        }
    ],
    "non_functional_requirements": [
        {
            "id": "NFR-001",
            "title": "Rendimiento / Seguridad / etc.",
            "description": "Descripción"
        }
    ],
    "technical_constraints": ["restricción 1"],
    "suggested_technologies": {
        "language": "python",
        "frameworks": ["framework1"],
        "libraries": ["lib1"],
        "tools": ["tool1"]
    },
    "data_model": {
        "entities": [
            {
                "name": "Entity",
                "attributes": ["attr1: type", "attr2: type"],
                "relationships": ["relación con otra entidad"]
            }
        ]
    },
    "user_stories": [
        "Como [rol], quiero [acción], para [beneficio]"
    ],
    "risks": ["riesgo potencial 1"],
    "estimated_effort": "small | medium | large | extra-large"
}
"""


class RequirementsAnalysisSkill:
    """
    Skill para analizar y estructurar requisitos de software.

    Capacidades:
      - Transformar lenguaje natural en requisitos técnicos
      - Identificar requisitos funcionales y no funcionales
      - Sugerir tecnologías apropiadas
      - Detectar ambigüedades y riesgos
      - Generar historias de usuario
      - Estimar esfuerzo
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        logger.info("Skill 'Análisis de Requisitos' inicializada")

    async def analyze(
        self,
        description: str,
        preferred_language: str = "python",
        constraints: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Analizar una descripción y generar requisitos estructurados.

        Args:
            description: Descripción del proyecto en lenguaje natural
            preferred_language: Lenguaje de programación preferido
            constraints: Restricciones adicionales del usuario

        Returns:
            Dict con requisitos estructurados
        """
        logger.info(f"Analizando requisitos: {description[:80]}...")

        prompt = self._build_prompt(description, preferred_language, constraints)

        raw = await self.llm.generate(
            prompt=prompt,
            system=ANALYSIS_SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=3000,
        )

        requirements = self._parse_requirements(raw)

        # Validar y enriquecer
        requirements = self._validate_requirements(requirements, preferred_language)

        logger.info(
            f"Análisis completado: {len(requirements.get('functional_requirements', []))} "
            f"requisitos funcionales, "
            f"{len(requirements.get('non_functional_requirements', []))} no funcionales"
        )

        return requirements

    def _build_prompt(
        self,
        description: str,
        language: str,
        constraints: Optional[List[str]],
    ) -> str:
        """Construir prompt para análisis."""
        prompt = f"""Analiza la siguiente descripción de proyecto y genera requisitos técnicos detallados:

DESCRIPCIÓN DEL PROYECTO:
{description}

LENGUAJE PREFERIDO: {language}
"""

        if constraints:
            prompt += f"\nRESTRICCIONES ADICIONALES:\n"
            for c in constraints:
                prompt += f"- {c}\n"

        prompt += "\nGenera los requisitos en formato JSON como se especificó."
        return prompt

    def _parse_requirements(self, raw: str) -> Dict[str, Any]:
        """Parsear requisitos desde la respuesta del LLM."""
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"Error parseando requisitos: {e}")

        return self._default_requirements()

    def _validate_requirements(
        self, req: Dict[str, Any], language: str
    ) -> Dict[str, Any]:
        """Validar y completar requisitos."""
        # Asegurar que tiene requisitos funcionales
        if not req.get("functional_requirements"):
            req["functional_requirements"] = [
                {
                    "id": "FR-001",
                    "title": "Funcionalidad principal",
                    "description": "Implementar la funcionalidad principal descrita",
                    "priority": "must",
                    "acceptance_criteria": ["El programa ejecuta sin errores"],
                }
            ]

        # Asegurar tecnologías sugeridas
        if not req.get("suggested_technologies"):
            req["suggested_technologies"] = {
                "language": language,
                "frameworks": [],
                "libraries": [],
                "tools": [],
            }

        # Asegurar esfuerzo estimado
        if not req.get("estimated_effort"):
            num_reqs = len(req.get("functional_requirements", []))
            if num_reqs <= 3:
                req["estimated_effort"] = "small"
            elif num_reqs <= 7:
                req["estimated_effort"] = "medium"
            elif num_reqs <= 15:
                req["estimated_effort"] = "large"
            else:
                req["estimated_effort"] = "extra-large"

        return req

    def _default_requirements(self) -> Dict[str, Any]:
        """Requisitos mínimos por defecto."""
        return {
            "project_type": "script",
            "functional_requirements": [],
            "non_functional_requirements": [],
            "technical_constraints": [],
            "suggested_technologies": {"language": "python"},
            "user_stories": [],
            "risks": [],
            "estimated_effort": "medium",
        }

    async def clarify_ambiguities(
        self, description: str, requirements: Dict[str, Any]
    ) -> List[str]:
        """
        Identificar ambigüedades que requieren clarificación.

        Returns:
            Lista de preguntas para el usuario
        """
        prompt = f"""Dado el proyecto descrito como:
"{description}"

Y los siguientes requisitos técnicos:
{json.dumps(requirements, indent=2, ensure_ascii=False)}

Identifica posibles ambigüedades o información faltante que necesite
clarificación del usuario. Lista las preguntas más importantes.
Responde SOLO con una lista JSON de strings (preguntas)."""

        raw = await self.llm.generate(prompt=prompt, temperature=0.4)

        try:
            start = raw.find("[")
            end = raw.rfind("]") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

        return []
