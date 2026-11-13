"""
agents/builder_agent.py - Agente Constructor.

AGENTE 2: El Constructor
========================
Responsable de generar el código fuente real, creando cada archivo
del proyecto según el plan definido por el Planificador. Utiliza
tanto Ollama directamente como OpenCode CLI para la generación.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

from agents.base_agent import AgentContext, BaseAgent
from core.event_bus import Event, EventType

logger = logging.getLogger(__name__)

BUILDER_SYSTEM_PROMPT = """Eres un programador experto de nivel senior.
Tu tarea es generar código fuente de alta calidad, completo y funcional.

REGLAS ESTRICTAS:
1. Genera SOLO código, sin explicaciones en prosa
2. El código debe ser completo y ejecutable (no uses placeholders como '...' o 'pass')
3. Incluye imports necesarios
4. Incluye docstrings y comentarios en español
5. Maneja errores con try/except apropiado
6. Sigue PEP 8 para Python, convenciones estándar para otros lenguajes
7. Incluye type hints cuando sea aplicable
8. El código debe ser modular y testeable
"""


class BuilderAgent(BaseAgent):
    """
    Agente Constructor - Genera código fuente para cada archivo del plan.

    Responsabilidades:
      - Crear la estructura de directorios
      - Generar cada archivo de código
      - Crear archivos de configuración
      - Generar README.md
      - Usar OpenCode para generación avanzada
    """

    def __init__(self, event_bus, llm_client, opencode):
        super().__init__(
            name="builder",
            role="Programador Senior / Constructor",
            event_bus=event_bus,
            llm_client=llm_client,
            opencode=opencode,
        )

    def _setup_subscriptions(self) -> None:
        """Suscribirse a eventos relevantes."""
        self.event_bus.subscribe(EventType.PLAN_GENERATED, self._on_plan_ready)
        self.event_bus.subscribe(
            EventType.REVIEW_ISSUES_FOUND, self._on_fix_requested
        )

    async def _on_plan_ready(self, event: Event) -> None:
        """Handler cuando el plan está listo."""
        logger.info(f"[{self.name}] Plan recibido, listo para construir")

    async def _on_fix_requested(self, event: Event) -> None:
        """Handler cuando se solicitan correcciones."""
        logger.info(f"[{self.name}] Correcciones solicitadas")

    async def execute(self, context: AgentContext) -> AgentContext:
        """
        Construir el proyecto completo basándose en el plan.

        1. Crear estructura de directorios
        2. Generar cada archivo según el plan
        3. Crear archivos auxiliares (README, config, etc.)
        4. Notificar progreso via EventBus
        """
        plan = context.plan
        if not plan:
            raise ValueError("No hay plan de desarrollo disponible")

        # Crear directorio del proyecto
        project_dir = Path(context.project_dir or "./output") / context.project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        context.project_dir = str(project_dir)

        await self.event_bus.emit(
            EventType.BUILD_STARTED,
            {"project_dir": str(project_dir), "files_count": len(plan.get("files", []))},
            source=self.name,
            correlation_id=context.project_id,
        )

        await self._log_action("build_started", {"project_dir": str(project_dir)})

        # Generar archivos
        files = plan.get("files", [])
        files_sorted = sorted(files, key=lambda f: f.get("priority", 99))
        generated_context = ""  # Contexto acumulado para coherencia

        for i, file_spec in enumerate(files_sorted):
            filepath = file_spec["path"]
            description = file_spec["description"]

            logger.info(
                f"[{self.name}] Generando ({i+1}/{len(files_sorted)}): {filepath}"
            )

            try:
                code = await self._generate_file(
                    filepath=filepath,
                    description=description,
                    context=context,
                    accumulated_context=generated_context,
                    project_dir=project_dir,
                )

                # Escribir archivo
                full_path = project_dir / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(code, encoding="utf-8")

                context.files_created.append(filepath)
                generated_context += f"\n--- {filepath} ---\n{code[:500]}\n"

                await self.event_bus.emit(
                    EventType.BUILD_FILE_CREATED,
                    {"file": filepath, "size": len(code)},
                    source=self.name,
                    correlation_id=context.project_id,
                )

                await self._log_action(
                    "file_created", {"file": filepath, "size": len(code)}
                )

            except Exception as e:
                logger.error(f"Error generando {filepath}: {e}")
                await self._log_action(
                    "file_error", {"file": filepath, "error": str(e)}
                )

        # Generar README si está configurado
        if plan.get("files") and not any(
            f["path"].lower() == "readme.md" for f in files
        ):
            await self._generate_readme(context, project_dir)

        # Generar requirements/dependencies
        await self._generate_dependencies(context, project_dir)

        await self.event_bus.emit(
            EventType.BUILD_COMPLETED,
            {
                "files_created": context.files_created,
                "project_dir": str(project_dir),
            },
            source=self.name,
            correlation_id=context.project_id,
        )

        return context

    async def _generate_file(
        self,
        filepath: str,
        description: str,
        context: AgentContext,
        accumulated_context: str,
        project_dir: Path,
    ) -> str:
        """
        Generar el contenido de un archivo.

        Intenta primero con OpenCode, y si falla, usa Ollama directamente.
        """
        # Intentar con OpenCode primero
        try:
            opencode_result = await self.opencode.generate_file(
                filepath=filepath,
                description=description,
                project_dir=str(project_dir),
            )

            if opencode_result:
                generated_file = project_dir / filepath
                if generated_file.exists():
                    code = generated_file.read_text(encoding="utf-8")
                    if code.strip():
                        logger.info(f"Archivo generado con OpenCode: {filepath}")
                        return code
        except Exception as e:
            logger.debug(f"OpenCode falló para {filepath}, usando Ollama: {e}")

        # Fallback: generar con Ollama directamente
        prompt = self._build_file_prompt(
            filepath, description, context, accumulated_context
        )

        code = await self.llm.generate(
            prompt=prompt,
            system=BUILDER_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )

        # Limpiar el código (quitar bloques markdown si existen)
        code = self._clean_code(code, filepath)
        return code

    def _build_file_prompt(
        self,
        filepath: str,
        description: str,
        context: AgentContext,
        accumulated: str,
    ) -> str:
        """Construir prompt para generación de archivo."""
        prompt = f"""Genera el archivo '{filepath}' para el proyecto '{context.project_name}'.

DESCRIPCIÓN DEL PROYECTO: {context.description}
LENGUAJE: {context.language}

DESCRIPCIÓN DEL ARCHIVO: {description}

ARQUITECTURA: {context.plan.get('architecture', {}).get('description', 'modular')}
"""

        if accumulated:
            prompt += f"""
ARCHIVOS YA GENERADOS (para mantener coherencia):
{accumulated}
"""

        prompt += """
Genera el código COMPLETO del archivo. No uses placeholders.
El código debe ser funcional e integrable con el resto del proyecto."""

        return prompt

    def _clean_code(self, code: str, filepath: str) -> str:
        """Limpiar código removiendo wrappers markdown."""
        lines = code.strip().split("\n")

        # Remover bloques ```language ... ```
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        return "\n".join(lines)

    async def _generate_readme(self, context: AgentContext, project_dir: Path) -> None:
        """Generar README.md del proyecto."""
        plan = context.plan
        prompt = f"""Genera un README.md completo para el proyecto '{context.project_name}'.

Descripción: {context.description}
Lenguaje: {context.language}
Arquitectura: {plan.get('architecture', {}).get('description', '')}
Dependencias: {', '.join(plan.get('dependencies', []))}
Archivos: {', '.join(context.files_created)}

Incluye secciones: Descripción, Instalación, Uso, Estructura, Contribución, Licencia.
Genera en formato Markdown."""

        readme = await self.llm.generate(prompt=prompt, temperature=0.4)
        readme_path = project_dir / "README.md"
        readme_path.write_text(readme, encoding="utf-8")
        context.files_created.append("README.md")
        logger.info("README.md generado")

    async def _generate_dependencies(
        self, context: AgentContext, project_dir: Path
    ) -> None:
        """Generar archivo de dependencias según el lenguaje."""
        deps = context.plan.get("dependencies", [])
        language = context.language.lower()

        if language == "python":
            req_path = project_dir / "requirements.txt"
            if deps:
                req_path.write_text("\n".join(deps) + "\n", encoding="utf-8")
            else:
                req_path.write_text("# Dependencias del proyecto\n", encoding="utf-8")
            context.files_created.append("requirements.txt")

        elif language in ("javascript", "typescript"):
            pkg = {
                "name": context.project_name,
                "version": "1.0.0",
                "description": context.description,
                "dependencies": {d: "latest" for d in deps},
            }
            import json
            pkg_path = project_dir / "package.json"
            pkg_path.write_text(json.dumps(pkg, indent=2), encoding="utf-8")
            context.files_created.append("package.json")
