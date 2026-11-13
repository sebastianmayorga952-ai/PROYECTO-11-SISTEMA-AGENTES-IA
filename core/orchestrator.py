"""
core/orchestrator.py - Orquestador del sistema agéntico.

El Orquestador es el componente central que coordina el flujo de trabajo
completo: desde la recepción del request hasta la entrega del proyecto
final. Gestiona los agentes, skills, y el ciclo de iteración.
"""

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from agents.base_agent import AgentContext
from agents.builder_agent import BuilderAgent
from agents.planner_agent import PlannerAgent
from agents.reviewer_agent import ReviewerAgent
from core.config import SystemConfig, load_config, setup_logging
from core.event_bus import EventBus, EventType
from core.llm_client import OllamaClient
from core.opencode_bridge import OpenCodeBridge
from skills.auto_testing import AutoTestingSkill
from skills.code_generation import CodeGenerationSkill
from skills.documentation import DocumentationSkill
from skills.requirements_analysis import RequirementsAnalysisSkill

logger = logging.getLogger(__name__)
console = Console()


class Orchestrator:
    """
    Orquestador principal del sistema agéntico.

    Flujo de trabajo:
      1. Recibe descripción del proyecto
      2. Skill de Análisis de Requisitos
      3. Agente Planificador genera plan
      4. Agente Constructor genera código (usando Skill de Generación)
      5. Skill de Testing genera y ejecuta tests
      6. Agente Revisor evalúa la calidad
      7. Si se aprueba → Skill de Documentación
      8. Si se rechaza → volver al paso 4 con correcciones
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or load_config()
        setup_logging(self.config.logging)

        # Infraestructura
        self.event_bus = EventBus()
        self.llm = OllamaClient(
            base_url=self.config.llm.base_url,
            default_model=self.config.llm.model,
            timeout=self.config.llm.timeout,
        )
        self.opencode = OpenCodeBridge(
            binary=self.config.opencode.binary,
            workspace=self.config.opencode.workspace,
            provider=self.config.llm.provider,
            model=self.config.llm.model,
        )

        # Agentes
        self.planner = PlannerAgent(self.event_bus, self.llm, self.opencode)
        self.builder = BuilderAgent(self.event_bus, self.llm, self.opencode)
        self.reviewer = ReviewerAgent(self.event_bus, self.llm, self.opencode)

        # Skills
        self.requirements_skill = RequirementsAnalysisSkill(self.llm)
        self.codegen_skill = CodeGenerationSkill(self.llm, self.opencode)
        self.testing_skill = AutoTestingSkill(self.llm, self.opencode)
        self.docs_skill = DocumentationSkill(self.llm)

        # Estado
        self._max_iterations = self.config.agents.retry_attempts + 1

        logger.info("Orquestador inicializado")

    async def create_project(
        self,
        description: str,
        language: str = "python",
        output_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Crear un proyecto completo a partir de una descripción.

        Args:
            description: Descripción del proyecto en lenguaje natural
            language: Lenguaje de programación
            output_dir: Directorio de salida

        Returns:
            Dict con el resultado completo del proyecto
        """
        start_time = time.time()
        output_dir = output_dir or self.config.output.base_dir

        console.print(
            Panel(
                f"[bold cyan]🚀 Iniciando generación de proyecto[/]\n\n"
                f"[white]{description}[/]\n\n"
                f"Lenguaje: [green]{language}[/]",
                title="OpenCode Agentic System",
                border_style="cyan",
            )
        )

        # Crear contexto
        context = AgentContext(
            description=description,
            language=language,
            project_dir=output_dir,
        )

        try:
            # ── FASE 1: Análisis de Requisitos ──────────────────
            console.print("\n[bold yellow]📋 Fase 1: Análisis de Requisitos[/]")
            requirements = await self.requirements_skill.analyze(
                description=description,
                preferred_language=language,
            )
            context.metadata["requirements"] = requirements
            console.print(
                f"  ✅ {len(requirements.get('functional_requirements', []))} "
                f"requisitos identificados | "
                f"Esfuerzo: {requirements.get('estimated_effort', 'N/A')}"
            )

            # ── FASE 2: Planificación ───────────────────────────
            console.print("\n[bold yellow]🗺️  Fase 2: Planificación[/]")
            context = await self.planner.run(context)
            plan = context.plan
            console.print(
                f"  ✅ Plan generado: {len(plan.get('files', []))} archivos, "
                f"{len(plan.get('tasks', []))} tareas | "
                f"Complejidad: {plan.get('estimated_complexity', 'N/A')}"
            )

            # ── FASE 3: Construcción ────────────────────────────
            iteration = 0
            approved = False

            while iteration < self._max_iterations and not approved:
                iteration += 1
                console.print(
                    f"\n[bold yellow]🔨 Fase 3: Construcción "
                    f"(iteración {iteration}/{self._max_iterations})[/]"
                )

                context = await self.builder.run(context)
                console.print(
                    f"  ✅ {len(context.files_created)} archivos generados"
                )

                # ── FASE 4: Testing ─────────────────────────────
                console.print(f"\n[bold yellow]🧪 Fase 4: Testing Automático[/]")
                test_results = await self.testing_skill.generate_project_tests(
                    project_dir=context.project_dir,
                    files=context.files_created,
                    language=context.language,
                )
                context.files_created.extend(test_results.keys())
                console.print(
                    f"  ✅ {len(test_results)} archivos de tests generados"
                )

                # ── FASE 5: Revisión ────────────────────────────
                console.print(f"\n[bold yellow]🔍 Fase 5: Revisión de Código[/]")
                context = await self.reviewer.run(context)
                review = context.review_results
                score = review.get("overall_score", 0)
                approved = review.get("approved", False)

                if approved:
                    console.print(
                        f"  ✅ [green]APROBADO[/] - Puntuación: {score}/10"
                    )
                else:
                    console.print(
                        f"  ⚠️  [yellow]CORRECCIONES NECESARIAS[/] - "
                        f"Puntuación: {score}/10 | "
                        f"Issues: {review.get('total_issues', 0)}"
                    )

            # ── FASE 6: Documentación ───────────────────────────
            console.print(f"\n[bold yellow]📝 Fase 6: Documentación[/]")
            docs = await self.docs_skill.generate_full_documentation(
                project_dir=context.project_dir,
                project_name=context.project_name,
                description=description,
                language=language,
                files=context.files_created,
                plan=plan,
            )
            context.files_created.extend(docs.keys())
            console.print(f"  ✅ {len(docs)} documentos generados")

            # ── RESULTADO FINAL ─────────────────────────────────
            elapsed = time.time() - start_time
            result = self._build_result(context, elapsed, approved)
            self._print_summary(result)

            # Publicar evento final
            await self.event_bus.emit(
                EventType.PROJECT_COMPLETED,
                result,
                source="orchestrator",
                correlation_id=context.project_id,
            )

            return result

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Error en generación: {e}", exc_info=True)

            await self.event_bus.emit(
                EventType.PROJECT_FAILED,
                {"error": str(e)},
                source="orchestrator",
                correlation_id=context.project_id,
            )

            console.print(f"\n[bold red]❌ Error: {e}[/]")
            return {
                "success": False,
                "error": str(e),
                "elapsed_seconds": round(elapsed, 1),
                "context": context.to_dict(),
            }

        finally:
            await self.llm.close()

    def _build_result(
        self, context: AgentContext, elapsed: float, approved: bool
    ) -> Dict[str, Any]:
        """Construir resultado final."""
        return {
            "success": True,
            "approved": approved,
            "project_id": context.project_id,
            "project_name": context.project_name,
            "project_dir": context.project_dir,
            "language": context.language,
            "files_created": context.files_created,
            "files_count": len(context.files_created),
            "review_score": context.review_results.get("overall_score", 0),
            "elapsed_seconds": round(elapsed, 1),
            "agents_used": ["planner", "builder", "reviewer"],
            "skills_used": [
                "requirements_analysis",
                "code_generation",
                "auto_testing",
                "documentation",
            ],
        }

    def _print_summary(self, result: Dict[str, Any]) -> None:
        """Imprimir resumen final."""
        table = Table(title="📊 Resumen del Proyecto", border_style="green")
        table.add_column("Propiedad", style="cyan", no_wrap=True)
        table.add_column("Valor", style="white")

        table.add_row("Proyecto", result["project_name"])
        table.add_row("Directorio", result["project_dir"])
        table.add_row("Lenguaje", result["language"])
        table.add_row("Archivos creados", str(result["files_count"]))
        table.add_row(
            "Puntuación",
            f"{result['review_score']}/10 {'✅' if result['approved'] else '⚠️'}",
        )
        table.add_row("Tiempo", f"{result['elapsed_seconds']}s")
        table.add_row("Agentes", ", ".join(result["agents_used"]))
        table.add_row("Skills", ", ".join(result["skills_used"]))

        console.print()
        console.print(table)
        console.print(
            f"\n[bold green]✅ Proyecto generado en:[/] {result['project_dir']}\n"
        )
