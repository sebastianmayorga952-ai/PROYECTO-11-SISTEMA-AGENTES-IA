"""
skills/code_generation.py - Skill de Generación de Código.

SKILL 2: Generación de Código
==============================
Genera código fuente de alta calidad para múltiples lenguajes,
soportando diferentes paradigmas (OOP, funcional, etc.) y
patrones de diseño.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.llm_client import OllamaClient
from core.opencode_bridge import OpenCodeBridge

logger = logging.getLogger(__name__)


# Templates de prompt por tipo de archivo
FILE_PROMPTS = {
    "module": """Genera un módulo {language} llamado '{name}'.
Funcionalidad: {description}
Requisitos:
- Código modular y reutilizable
- Documentación completa (docstrings)
- Type hints
- Manejo de errores""",

    "class": """Genera una clase {language} llamada '{name}'.
Responsabilidad: {description}
Requisitos:
- Encapsulación apropiada
- Métodos bien definidos
- Constructor claro
- Herencia solo si es necesario""",

    "api": """Genera un endpoint/controlador API en {language}.
Recurso: {name}
Funcionalidad: {description}
Requisitos:
- Validación de entrada
- Manejo de errores HTTP
- Respuestas JSON estructuradas
- Documentación de la API""",

    "script": """Genera un script {language} ejecutable.
Nombre: {name}
Funcionalidad: {description}
Requisitos:
- Argumentos de línea de comandos
- Logging apropiado
- Código principal bajo if __name__ == '__main__'
- Mensajes de ayuda""",

    "test": """Genera tests unitarios en {language} para el módulo '{name}'.
Módulo a testear: {description}
Requisitos:
- Cobertura de casos normales
- Cobertura de casos edge
- Mocks cuando sea necesario
- Assertions descriptivos""",

    "config": """Genera un archivo de configuración '{name}'.
Propósito: {description}
Formato apropiado para el tipo de archivo.""",
}

CODE_SYSTEM_PROMPT = """Eres un programador experto que genera código de producción.

REGLAS ABSOLUTAS:
1. Genera SOLO código fuente, sin explicaciones en prosa
2. El código debe ser 100% completo y funcional
3. NO uses placeholders, TODO, o pass (excepto en abstract methods)
4. Incluye TODOS los imports necesarios
5. Incluye docstrings y comentarios en español
6. Sigue las convenciones del lenguaje estrictamente
7. Maneja todos los errores posibles
8. El código debe poder ejecutarse directamente
"""


class CodeGenerationSkill:
    """
    Skill para generar código fuente de alta calidad.

    Capacidades:
      - Generación multi-lenguaje (Python, JS, TS, Go, etc.)
      - Soporte para diferentes tipos de archivos
      - Generación con contexto (archivos relacionados)
      - Integración con OpenCode para generación avanzada
      - Limpieza y formateo automático
    """

    def __init__(self, llm_client: OllamaClient, opencode: OpenCodeBridge):
        self.llm = llm_client
        self.opencode = opencode
        logger.info("Skill 'Generación de Código' inicializada")

    async def generate(
        self,
        filepath: str,
        description: str,
        language: str = "python",
        file_type: str = "module",
        context_files: Optional[Dict[str, str]] = None,
        project_dir: Optional[str] = None,
    ) -> str:
        """
        Generar código para un archivo.

        Args:
            filepath: Ruta del archivo a crear
            description: Qué debe hacer el código
            language: Lenguaje de programación
            file_type: Tipo de archivo (module, class, api, script, test, config)
            context_files: Archivos existentes para coherencia {path: content}
            project_dir: Directorio del proyecto para OpenCode

        Returns:
            Código fuente generado
        """
        logger.info(f"Generando código: {filepath} ({file_type}/{language})")

        name = Path(filepath).stem

        # Intentar con OpenCode primero si hay un proyecto
        if project_dir:
            code = await self._try_opencode(filepath, description, project_dir)
            if code:
                return code

        # Generar con Ollama
        prompt = self._build_prompt(name, description, language, file_type, context_files)

        code = await self.llm.generate(
            prompt=prompt,
            system=CODE_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )

        # Limpiar y validar
        code = self._clean_output(code, language)
        return code

    async def generate_project_files(
        self,
        files_spec: List[Dict[str, Any]],
        language: str,
        project_description: str,
        output_dir: str,
    ) -> Dict[str, str]:
        """
        Generar múltiples archivos de un proyecto.

        Args:
            files_spec: Lista de especificaciones de archivos
            language: Lenguaje principal
            project_description: Descripción del proyecto
            output_dir: Directorio de salida

        Returns:
            Dict {filepath: code} de archivos generados
        """
        results = {}
        context_files = {}
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Ordenar por prioridad
        sorted_files = sorted(files_spec, key=lambda f: f.get("priority", 99))

        for spec in sorted_files:
            filepath = spec["path"]
            description = spec.get("description", "")
            file_type = self._infer_file_type(filepath, description)

            try:
                code = await self.generate(
                    filepath=filepath,
                    description=f"{description}\n\nContexto del proyecto: {project_description}",
                    language=language,
                    file_type=file_type,
                    context_files=context_files,
                    project_dir=output_dir,
                )

                results[filepath] = code
                context_files[filepath] = code[:500]  # Mantener contexto limitado

                # Guardar archivo
                full_path = output_path / filepath
                full_path.parent.mkdir(parents=True, exist_ok=True)
                full_path.write_text(code, encoding="utf-8")

                logger.info(f"Archivo generado: {filepath} ({len(code)} chars)")

            except Exception as e:
                logger.error(f"Error generando {filepath}: {e}")
                results[filepath] = f"# Error generando archivo: {e}\n"

        return results

    async def _try_opencode(
        self, filepath: str, description: str, project_dir: str
    ) -> Optional[str]:
        """Intentar generar con OpenCode."""
        try:
            result = await self.opencode.generate_file(
                filepath=filepath,
                description=description,
                project_dir=project_dir,
            )

            if result:
                full_path = Path(project_dir) / filepath
                if full_path.exists():
                    code = full_path.read_text(encoding="utf-8")
                    if code.strip():
                        logger.info(f"Generado con OpenCode: {filepath}")
                        return code
        except Exception as e:
            logger.debug(f"OpenCode no disponible: {e}")

        return None

    def _build_prompt(
        self,
        name: str,
        description: str,
        language: str,
        file_type: str,
        context_files: Optional[Dict[str, str]],
    ) -> str:
        """Construir prompt para generación."""
        template = FILE_PROMPTS.get(file_type, FILE_PROMPTS["module"])
        prompt = template.format(
            name=name, description=description, language=language
        )

        if context_files:
            prompt += "\n\nARCHIVOS EXISTENTES DEL PROYECTO (para coherencia):\n"
            for path, content in list(context_files.items())[-5:]:  # Últimos 5
                prompt += f"\n--- {path} ---\n{content}\n"

        return prompt

    def _infer_file_type(self, filepath: str, description: str) -> str:
        """Inferir el tipo de archivo por su nombre y descripción."""
        name = Path(filepath).stem.lower()
        desc_lower = description.lower()

        if "test" in name or "spec" in name:
            return "test"
        if name in ("main", "app", "cli", "run", "__main__"):
            return "script"
        if any(kw in desc_lower for kw in ("api", "endpoint", "route", "controller")):
            return "api"
        if any(kw in desc_lower for kw in ("clase", "class", "modelo", "model")):
            return "class"
        if Path(filepath).suffix in (".json", ".yaml", ".yml", ".toml", ".ini", ".env"):
            return "config"

        return "module"

    def _clean_output(self, code: str, language: str) -> str:
        """Limpiar la salida del LLM."""
        lines = code.strip().split("\n")

        # Remover marcadores markdown
        if lines and re.match(r"^```\w*$", lines[0]):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        # Remover líneas de explicación al inicio
        while lines and not self._is_code_line(lines[0], language):
            if lines[0].strip().startswith(("#!", "//", "/*", '"""', "'''", "#")):
                break  # Es un comentario de código, mantener
            if lines[0].strip() == "":
                break  # Línea vacía, mantener
            lines.pop(0)

        return "\n".join(lines)

    def _is_code_line(self, line: str, language: str) -> bool:
        """Determinar si una línea es código."""
        stripped = line.strip()
        if not stripped:
            return True  # Líneas vacías son válidas

        code_indicators = {
            "python": ["import ", "from ", "def ", "class ", "#", '"""', "'''", "@"],
            "javascript": ["import ", "const ", "let ", "var ", "function ", "//", "/*", "export "],
            "typescript": ["import ", "const ", "let ", "var ", "function ", "//", "/*", "export ", "interface "],
        }

        indicators = code_indicators.get(language.lower(), [])
        return any(stripped.startswith(ind) for ind in indicators) or stripped.startswith("#!")
