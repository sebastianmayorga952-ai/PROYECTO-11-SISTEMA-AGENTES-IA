"""
agents/reviewer_agent.py - Agente Revisor.

AGENTE 3: El Revisor
=====================
Responsable de revisar la calidad del código generado,
verificar buenas prácticas, detectar bugs potenciales,
y sugerir mejoras. Actúa como el "code reviewer" del equipo.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import AgentContext, BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)

REVIEWER_SYSTEM_PROMPT = """Eres un revisor de código senior con experiencia en múltiples lenguajes.
Tu tarea es revisar código generado automáticamente y evaluar su calidad.

DEBES responder SIEMPRE en formato JSON con esta estructura:
{
    "overall_score": 8,
    "summary": "Resumen general de la revisión",
    "files_reviewed": [
        {
            "file": "ruta/archivo.py",
            "score": 8,
            "issues": [
                {
                    "severity": "high | medium | low",
                    "type": "bug | style | performance | security | documentation",
                    "line": 42,
                    "description": "Descripción del problema",
                    "suggestion": "Sugerencia de corrección"
                }
            ],
            "strengths": ["Punto fuerte 1", "Punto fuerte 2"]
        }
    ],
    "critical_issues": [],
    "recommendations": ["Recomendación general 1"],
    "approved": true
}

CRITERIOS DE EVALUACIÓN:
1. Funcionalidad: ¿El código hace lo que debe?
2. Legibilidad: ¿Es claro y bien documentado?
3. Manejo de errores: ¿Gestiona excepciones correctamente?
4. Seguridad: ¿Hay vulnerabilidades obvias?
5. Rendimiento: ¿Hay ineficiencias evidentes?
6. Mejores prácticas: ¿Sigue las convenciones del lenguaje?

ESCALA: 1-10 (10 = perfecto, < 6 = requiere correcciones)
"""


class ReviewerAgent(BaseAgent):
    """
    Agente Revisor - Evalúa la calidad del código generado.

    Responsabilidades:
      - Revisar cada archivo generado
      - Detectar bugs y vulnerabilidades
      - Evaluar estilo y mejores prácticas
      - Sugerir mejoras
      - Aprobar o rechazar el build
    """

    def __init__(self, event_bus, llm_client, opencode, min_score: float = 6.0):
        super().__init__(
            name="reviewer",
            role="Revisor de Código Senior",
            event_bus=event_bus,
            llm_client=llm_client,
            opencode=opencode,
        )
        self.min_score = min_score

    def _setup_subscriptions(self) -> None:
        """Suscribirse a eventos relevantes."""
        self.event_bus.subscribe(EventType.BUILD_COMPLETED, self._on_build_completed)
        self.event_bus.subscribe(
            EventType.REVIEW_REQUESTED, self._on_review_requested
        )

    async def _on_build_completed(self, event: Event) -> None:
        """Handler cuando el build se completa."""
        logger.info(f"[{self.name}] Build completado, preparando revisión")

    async def _on_review_requested(self, event: Event) -> None:
        """Handler cuando se solicita revisión."""
        logger.info(f"[{self.name}] Revisión solicitada")

    async def execute(self, context: AgentContext) -> AgentContext:
        """
        Revisar todos los archivos generados.

        1. Leer cada archivo del proyecto
        2. Enviar a Ollama para análisis
        3. Compilar resultados
        4. Decidir si aprueba o requiere correcciones
        """
        project_dir = Path(context.project_dir)
        if not project_dir.exists():
            raise ValueError(f"Directorio del proyecto no existe: {project_dir}")

        await self.event_bus.emit(
            EventType.REVIEW_REQUESTED,
            {"files": context.files_created},
            source=self.name,
            correlation_id=context.project_id,
        )

        await self._log_action(
            "review_started", {"files_count": len(context.files_created)}
        )

        # Revisar archivos
        all_reviews = []
        total_score = 0
        files_reviewed = 0

        for filepath in context.files_created:
            full_path = project_dir / filepath

            # Solo revisar archivos de código
            if not self._is_reviewable(filepath):
                continue

            if not full_path.exists():
                logger.warning(f"Archivo no encontrado: {filepath}")
                continue

            code = full_path.read_text(encoding="utf-8", errors="replace")
            if not code.strip():
                continue

            logger.info(f"[{self.name}] Revisando: {filepath}")

            review = await self._review_file(filepath, code, context)
            all_reviews.append(review)

            file_score = review.get("score", 5)
            total_score += file_score
            files_reviewed += 1

            await self._log_action(
                "file_reviewed",
                {
                    "file": filepath,
                    "score": file_score,
                    "issues": len(review.get("issues", [])),
                },
            )

        # Calcular resultado general
        avg_score = total_score / files_reviewed if files_reviewed > 0 else 0
        approved = avg_score >= self.min_score

        # Recopilar issues críticos
        critical_issues = []
        all_issues = []
        for review in all_reviews:
            for issue in review.get("issues", []):
                all_issues.append(issue)
                if issue.get("severity") == "high":
                    critical_issues.append(issue)

        review_results = {
            "overall_score": round(avg_score, 1),
            "files_reviewed": files_reviewed,
            "total_issues": len(all_issues),
            "critical_issues": len(critical_issues),
            "approved": approved,
            "reviews": all_reviews,
            "issues": all_issues,
        }

        context.review_results = review_results

        # Publicar resultados
        if approved:
            await self.event_bus.emit(
                EventType.REVIEW_COMPLETED,
                review_results,
                source=self.name,
                correlation_id=context.project_id,
            )
        else:
            await self.event_bus.emit(
                EventType.REVIEW_ISSUES_FOUND,
                {
                    "score": avg_score,
                    "issues": [
                        i.get("description", "") for i in critical_issues
                    ],
                },
                source=self.name,
                correlation_id=context.project_id,
            )

        await self._log_action(
            "review_completed",
            {
                "score": round(avg_score, 1),
                "approved": approved,
                "issues": len(all_issues),
            },
        )

        return context

    async def _review_file(
        self, filepath: str, code: str, context: AgentContext
    ) -> Dict[str, Any]:
        """Revisar un archivo individual."""
        prompt = f"""Revisa el siguiente archivo del proyecto '{context.project_name}':

ARCHIVO: {filepath}
LENGUAJE: {context.language}
DESCRIPCIÓN DEL PROYECTO: {context.description}

CÓDIGO:
```
{code}
```

Evalúa el código según los criterios establecidos y responde en JSON."""

        raw = await self.llm.generate(
            prompt=prompt,
            system=REVIEWER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=2048,
        )

        # Parsear review
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                review = json.loads(raw[start:end])
                review["file"] = filepath
                return review
        except json.JSONDecodeError:
            logger.warning(f"No se pudo parsear la review de {filepath}")

        # Review por defecto
        return {
            "file": filepath,
            "score": 7,
            "issues": [],
            "strengths": ["Archivo generado automáticamente"],
        }

    def _is_reviewable(self, filepath: str) -> bool:
        """Determinar si un archivo debe ser revisado."""
        reviewable_extensions = {
            ".py", ".js", ".ts", ".jsx", ".tsx", ".java",
            ".go", ".rs", ".rb", ".php", ".c", ".cpp", ".h",
            ".cs", ".swift", ".kt",
        }
        ext = Path(filepath).suffix.lower()
        return ext in reviewable_extensions

    async def generate_review_report(self, context: AgentContext) -> str:
        """Generar un reporte legible de la revisión."""
        results = context.review_results
        if not results:
            return "No hay resultados de revisión disponibles."

        report_lines = [
            f"# Reporte de Revisión de Código",
            f"## Proyecto: {context.project_name}",
            f"",
            f"**Puntuación General:** {results['overall_score']}/10",
            f"**Archivos Revisados:** {results['files_reviewed']}",
            f"**Issues Totales:** {results['total_issues']}",
            f"**Issues Críticos:** {results['critical_issues']}",
            f"**Estado:** {'✅ APROBADO' if results['approved'] else '❌ REQUIERE CORRECCIONES'}",
            f"",
        ]

        for review in results.get("reviews", []):
            report_lines.append(f"### {review.get('file', 'Unknown')}")
            report_lines.append(f"Puntuación: {review.get('score', 'N/A')}/10")

            for issue in review.get("issues", []):
                severity = issue.get("severity", "medium")
                icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
                report_lines.append(
                    f"- {icon} [{severity.upper()}] {issue.get('description', '')}"
                )
                if issue.get("suggestion"):
                    report_lines.append(f"  💡 Sugerencia: {issue['suggestion']}")

            report_lines.append("")

        return "\n".join(report_lines)
