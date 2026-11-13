"""
skills/auto_testing.py - Skill de Testing Automático.

SKILL 3: Testing Automático
============================
Genera y ejecuta tests unitarios automáticamente para el código
generado, evaluando cobertura y reportando resultados.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.llm_client import OllamaClient
from core.opencode_bridge import OpenCodeBridge

logger = logging.getLogger(__name__)

TEST_SYSTEM_PROMPT = """Eres un ingeniero de QA experto especializado en testing automatizado.
Tu tarea es generar tests unitarios completos y robustos.

REGLAS:
1. Genera tests que cubran todos los caminos de ejecución
2. Incluye tests para casos edge y errores
3. Usa fixtures y mocks apropiadamente
4. Cada test debe tener un nombre descriptivo
5. Incluye docstrings explicando qué se testea
6. Usa assertions específicos (no solo assert True)
7. Organiza tests en clases lógicas

Para Python usa pytest. Para JavaScript usa jest. Para otros lenguajes usa
el framework de testing estándar.

Genera SOLO código de tests, sin explicaciones adicionales.
"""


class AutoTestingSkill:
    """
    Skill para generar y ejecutar tests automáticos.

    Capacidades:
      - Generar tests unitarios para código fuente
      - Generar tests de integración
      - Ejecutar tests via OpenCode o directamente
      - Calcular y reportar cobertura estimada
      - Sugerir tests faltantes
    """

    def __init__(self, llm_client: OllamaClient, opencode: OpenCodeBridge):
        self.llm = llm_client
        self.opencode = opencode
        logger.info("Skill 'Testing Automático' inicializada")

    async def generate_tests(
        self,
        source_file: str,
        source_code: str,
        language: str = "python",
        test_framework: Optional[str] = None,
    ) -> str:
        """
        Generar tests unitarios para un archivo de código.

        Args:
            source_file: Nombre del archivo fuente
            source_code: Código fuente a testear
            language: Lenguaje de programación
            test_framework: Framework de testing (pytest, jest, etc.)

        Returns:
            Código de tests generado
        """
        framework = test_framework or self._default_framework(language)

        logger.info(f"Generando tests para {source_file} ({framework})")

        prompt = f"""Genera tests unitarios completos para el siguiente código:

ARCHIVO: {source_file}
LENGUAJE: {language}
FRAMEWORK DE TESTING: {framework}

CÓDIGO FUENTE:
```{language}
{source_code}
```

Genera tests que cubran:
1. Todos los métodos/funciones públicas
2. Casos normales de uso
3. Casos edge (valores límite, listas vacías, None, etc.)
4. Manejo de errores y excepciones
5. Interacciones entre componentes (si aplica)

El archivo de test debe llamarse 'test_{Path(source_file).stem}.{self._test_extension(language)}'"""

        test_code = await self.llm.generate(
            prompt=prompt,
            system=TEST_SYSTEM_PROMPT,
            temperature=0.2,
            max_tokens=4096,
        )

        # Limpiar
        test_code = self._clean_test_code(test_code)
        return test_code

    async def generate_project_tests(
        self,
        project_dir: str,
        files: List[str],
        language: str = "python",
    ) -> Dict[str, str]:
        """
        Generar tests para múltiples archivos de un proyecto.

        Args:
            project_dir: Directorio del proyecto
            files: Lista de archivos a testear
            language: Lenguaje

        Returns:
            Dict {test_file: test_code}
        """
        project_path = Path(project_dir)
        test_dir = project_path / "tests"
        test_dir.mkdir(exist_ok=True)

        results = {}

        # Crear __init__.py para Python
        if language == "python":
            init_file = test_dir / "__init__.py"
            if not init_file.exists():
                init_file.write_text('"""Tests del proyecto."""\n', encoding="utf-8")

        for filepath in files:
            # Solo testear archivos de código
            if not self._is_testable(filepath):
                continue

            full_path = project_path / filepath
            if not full_path.exists():
                continue

            source_code = full_path.read_text(encoding="utf-8", errors="replace")
            if not source_code.strip():
                continue

            try:
                test_code = await self.generate_tests(
                    source_file=filepath,
                    source_code=source_code,
                    language=language,
                )

                test_filename = f"test_{Path(filepath).stem}{self._test_extension(language)}"
                test_path = test_dir / test_filename
                test_path.write_text(test_code, encoding="utf-8")

                results[f"tests/{test_filename}"] = test_code
                logger.info(f"Tests generados: tests/{test_filename}")

            except Exception as e:
                logger.error(f"Error generando tests para {filepath}: {e}")

        # Generar conftest.py para pytest
        if language == "python" and results:
            conftest = await self._generate_conftest(project_dir, files)
            conftest_path = test_dir / "conftest.py"
            conftest_path.write_text(conftest, encoding="utf-8")
            results["tests/conftest.py"] = conftest

        return results

    async def run_tests(
        self, project_dir: str, language: str = "python"
    ) -> Dict[str, Any]:
        """
        Ejecutar tests del proyecto.

        Returns:
            Resultados de la ejecución
        """
        logger.info(f"Ejecutando tests en {project_dir}")

        # Intentar con OpenCode
        try:
            result = await self.opencode.run_tests(project_dir=project_dir)
            return self._parse_test_results(result, language)
        except Exception as e:
            logger.debug(f"OpenCode no disponible para tests: {e}")

        # Resultado por defecto (tests no ejecutados)
        return {
            "executed": False,
            "message": "Tests generados pero no ejecutados. Ejecute manualmente.",
            "command": self._test_command(language),
        }

    async def estimate_coverage(
        self,
        source_code: str,
        test_code: str,
        language: str = "python",
    ) -> Dict[str, Any]:
        """
        Estimar la cobertura de tests usando el LLM.

        Returns:
            Estimación de cobertura
        """
        prompt = f"""Analiza el siguiente código fuente y sus tests para estimar la cobertura:

CÓDIGO FUENTE:
```
{source_code[:2000]}
```

TESTS:
```
{test_code[:2000]}
```

Responde en JSON:
{{
    "estimated_coverage_percent": 75,
    "covered_functions": ["func1", "func2"],
    "uncovered_functions": ["func3"],
    "missing_test_cases": [
        "Test para caso edge X",
        "Test para error Y"
    ],
    "quality_score": 7
}}"""

        raw = await self.llm.generate(prompt=prompt, temperature=0.2, max_tokens=1500)

        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass

        return {
            "estimated_coverage_percent": 50,
            "quality_score": 5,
            "missing_test_cases": ["Análisis de cobertura no disponible"],
        }

    async def _generate_conftest(self, project_dir: str, files: List[str]) -> str:
        """Generar conftest.py para pytest."""
        prompt = f"""Genera un archivo conftest.py para pytest con fixtures útiles
para un proyecto que contiene estos archivos: {', '.join(files)}

Incluye:
- Fixtures para datos de prueba
- Fixtures para configuración temporal
- Fixtures para cleanup
- Marcadores personalizados si es necesario

Genera SOLO código Python."""

        code = await self.llm.generate(
            prompt=prompt, system=TEST_SYSTEM_PROMPT, temperature=0.2
        )
        return self._clean_test_code(code)

    def _is_testable(self, filepath: str) -> bool:
        """Determinar si un archivo necesita tests."""
        name = Path(filepath).stem.lower()
        ext = Path(filepath).suffix.lower()

        # Ignorar archivos de test existentes
        if name.startswith("test_") or name.endswith("_test"):
            return False

        # Ignorar archivos de configuración
        if name in ("__init__", "conftest", "setup", "config"):
            return False

        testable_exts = {".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs"}
        return ext in testable_exts

    def _test_extension(self, language: str) -> str:
        """Obtener extensión de archivo de test."""
        extensions = {
            "python": ".py",
            "javascript": ".js",
            "typescript": ".ts",
            "go": "_test.go",
            "java": ".java",
            "rust": ".rs",
        }
        return extensions.get(language.lower(), ".py")

    def _default_framework(self, language: str) -> str:
        """Framework de testing por defecto."""
        frameworks = {
            "python": "pytest",
            "javascript": "jest",
            "typescript": "jest",
            "go": "testing",
            "java": "JUnit 5",
            "rust": "built-in #[cfg(test)]",
        }
        return frameworks.get(language.lower(), "pytest")

    def _test_command(self, language: str) -> str:
        """Comando para ejecutar tests."""
        commands = {
            "python": "pytest tests/ -v",
            "javascript": "npx jest",
            "typescript": "npx jest",
            "go": "go test ./...",
            "java": "mvn test",
            "rust": "cargo test",
        }
        return commands.get(language.lower(), "pytest tests/ -v")

    def _parse_test_results(
        self, result: Dict[str, Any], language: str
    ) -> Dict[str, Any]:
        """Parsear resultados de ejecución de tests."""
        stdout = result.get("stdout", "")
        returncode = result.get("returncode", -1)

        return {
            "executed": True,
            "passed": returncode == 0,
            "output": stdout[:2000],
            "returncode": returncode,
        }

    def _clean_test_code(self, code: str) -> str:
        """Limpiar código de tests."""
        lines = code.strip().split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)
