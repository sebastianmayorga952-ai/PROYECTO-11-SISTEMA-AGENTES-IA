"""
skills/documentation.py - Skill de Documentación Automática.

SKILL 4: Documentación Automática
===================================
Genera documentación completa para proyectos, incluyendo README,
guías de uso, documentación de API, y diagramas de arquitectura.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.llm_client import OllamaClient

logger = logging.getLogger(__name__)

DOC_SYSTEM_PROMPT = """Eres un technical writer experto.
Tu tarea es crear documentación clara, completa y profesional para proyectos de software.

REGLAS:
1. Usa Markdown para formateo
2. Sé claro y conciso
3. Incluye ejemplos de código cuando sea útil
4. Estructura la información de forma lógica
5. Usa un tono profesional pero accesible
6. Documenta tanto para usuarios como para desarrolladores
"""


class DocumentationSkill:
    """
    Skill para generar documentación automática de proyectos.

    Capacidades:
      - Generar README.md completo
      - Documentación de API / módulos
      - Guías de instalación y uso
      - Diagramas de arquitectura (Mermaid)
      - Changelog y contributing guidelines
      - Documentación inline (docstrings)
    """

    def __init__(self, llm_client: OllamaClient):
        self.llm = llm_client
        logger.info("Skill 'Documentación Automática' inicializada")

    async def generate_readme(
        self,
        project_name: str,
        description: str,
        language: str,
        files: List[str],
        plan: Dict[str, Any],
    ) -> str:
        """
        Generar README.md completo del proyecto.

        Returns:
            Contenido del README en Markdown
        """
        logger.info(f"Generando README para '{project_name}'")

        deps = plan.get("dependencies", [])
        arch = plan.get("architecture", {})
        tree = self._build_file_tree(files)

        prompt = f"""Genera un README.md completo y profesional para el siguiente proyecto:

NOMBRE: {project_name}
DESCRIPCIÓN: {description}
LENGUAJE: {language}
ARQUITECTURA: {arch.get('pattern', 'modular')} - {arch.get('description', '')}
DEPENDENCIAS: {', '.join(deps) if deps else 'Ninguna adicional'}

ESTRUCTURA DE ARCHIVOS:
{tree}

El README debe incluir:
1. **Título y badges** (build status, versión, licencia)
2. **Descripción** - Qué hace el proyecto y por qué es útil
3. **Características** - Lista de funcionalidades principales
4. **Requisitos Previos** - Qué se necesita para ejecutar
5. **Instalación** - Pasos detallados de instalación
6. **Uso** - Ejemplos de uso con código
7. **Estructura del Proyecto** - Árbol de archivos con descripción
8. **API / Referencia** - Descripción de módulos principales
9. **Testing** - Cómo ejecutar los tests
10. **Contribución** - Guía para contribuir
11. **Licencia** - MIT por defecto
12. **Autores** - Generado por OpenCode Agentic System

Formato: Markdown estándar de GitHub."""

        readme = await self.llm.generate(
            prompt=prompt,
            system=DOC_SYSTEM_PROMPT,
            temperature=0.4,
            max_tokens=4096,
        )

        return readme

    async def generate_api_docs(
        self,
        project_dir: str,
        files: List[str],
        language: str = "python",
    ) -> str:
        """
        Generar documentación de API/módulos.

        Returns:
            Documentación de API en Markdown
        """
        logger.info("Generando documentación de API")

        project_path = Path(project_dir)
        docs_parts = ["# Documentación de API\n"]

        for filepath in files:
            if not self._is_documentable(filepath):
                continue

            full_path = project_path / filepath
            if not full_path.exists():
                continue

            code = full_path.read_text(encoding="utf-8", errors="replace")
            if not code.strip():
                continue

            doc = await self._document_file(filepath, code, language)
            docs_parts.append(doc)

        return "\n\n---\n\n".join(docs_parts)

    async def generate_architecture_diagram(
        self,
        project_name: str,
        plan: Dict[str, Any],
        files: List[str],
    ) -> str:
        """
        Generar diagrama de arquitectura en Mermaid.

        Returns:
            Diagrama Mermaid en string
        """
        logger.info("Generando diagrama de arquitectura")

        prompt = f"""Genera un diagrama de arquitectura en formato Mermaid para este proyecto:

PROYECTO: {project_name}
ARQUITECTURA: {json.dumps(plan.get('architecture', {}), ensure_ascii=False)}
ARCHIVOS: {', '.join(files)}

Genera un diagrama tipo 'graph TD' o 'flowchart TD' que muestre:
- Los módulos/componentes principales
- Las relaciones y dependencias entre ellos
- El flujo de datos
- Los puntos de entrada

Responde SOLO con el código Mermaid (sin bloques ```mermaid```, solo el contenido)."""

        diagram = await self.llm.generate(
            prompt=prompt, temperature=0.3, max_tokens=1500
        )

        # Limpiar
        diagram = diagram.strip()
        if diagram.startswith("```"):
            lines = diagram.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            diagram = "\n".join(lines)

        return diagram

    async def generate_usage_guide(
        self,
        project_name: str,
        description: str,
        language: str,
        main_file: Optional[str] = None,
    ) -> str:
        """
        Generar guía de uso detallada.

        Returns:
            Guía de uso en Markdown
        """
        prompt = f"""Genera una guía de uso detallada para el proyecto '{project_name}'.

DESCRIPCIÓN: {description}
LENGUAJE: {language}
ARCHIVO PRINCIPAL: {main_file or 'main.py'}

La guía debe incluir:
1. Inicio rápido (Quick Start)
2. Configuración
3. Ejemplos de uso comunes
4. Opciones de línea de comandos (si aplica)
5. Preguntas frecuentes (FAQ)
6. Solución de problemas (Troubleshooting)

Formato: Markdown."""

        return await self.llm.generate(
            prompt=prompt, system=DOC_SYSTEM_PROMPT, temperature=0.4
        )

    async def generate_full_documentation(
        self,
        project_dir: str,
        project_name: str,
        description: str,
        language: str,
        files: List[str],
        plan: Dict[str, Any],
    ) -> Dict[str, str]:
        """
        Generar toda la documentación del proyecto.

        Returns:
            Dict {filename: content} de archivos de documentación
        """
        logger.info(f"Generando documentación completa para '{project_name}'")

        project_path = Path(project_dir)
        docs_dir = project_path / "docs"
        docs_dir.mkdir(exist_ok=True)

        docs = {}

        # 1. README.md
        readme = await self.generate_readme(
            project_name, description, language, files, plan
        )
        docs["README.md"] = readme

        # 2. Documentación de API
        api_docs = await self.generate_api_docs(project_dir, files, language)
        docs["docs/API.md"] = api_docs

        # 3. Diagrama de arquitectura
        diagram = await self.generate_architecture_diagram(
            project_name, plan, files
        )
        arch_doc = f"# Arquitectura de {project_name}\n\n```mermaid\n{diagram}\n```\n"
        docs["docs/ARCHITECTURE.md"] = arch_doc

        # 4. Guía de uso
        usage = await self.generate_usage_guide(
            project_name, description, language
        )
        docs["docs/USAGE.md"] = usage

        # Guardar archivos
        for filepath, content in docs.items():
            full_path = project_path / filepath
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding="utf-8")
            logger.info(f"Documentación generada: {filepath}")

        return docs

    async def _document_file(
        self, filepath: str, code: str, language: str
    ) -> str:
        """Documentar un archivo individual."""
        prompt = f"""Genera documentación técnica para el siguiente archivo:

ARCHIVO: {filepath}
LENGUAJE: {language}

CÓDIGO:
```{language}
{code[:3000]}
```

Genera documentación que incluya:
- Descripción general del módulo
- Lista de clases/funciones con descripción
- Parámetros y valores de retorno
- Ejemplos de uso

Formato: Markdown."""

        return await self.llm.generate(
            prompt=prompt, system=DOC_SYSTEM_PROMPT, temperature=0.3, max_tokens=2000
        )

    def _build_file_tree(self, files: List[str]) -> str:
        """Construir representación de árbol de archivos."""
        if not files:
            return "  (vacío)"

        lines = []
        for f in sorted(files):
            parts = Path(f).parts
            indent = "  " * (len(parts) - 1)
            lines.append(f"{indent}├── {parts[-1]}")

        return "\n".join(lines)

    def _is_documentable(self, filepath: str) -> bool:
        """Determinar si un archivo necesita documentación de API."""
        ext = Path(filepath).suffix.lower()
        name = Path(filepath).stem.lower()

        if name.startswith("test_") or name == "__init__":
            return False

        return ext in {".py", ".js", ".ts", ".java", ".go", ".rs"}
