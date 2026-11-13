"""
core/opencode_bridge.py - Puente de integración con OpenCode CLI.

Permite ejecutar comandos de OpenCode programáticamente, tanto en
modo interactivo (TUI) como en modo headless (opencode run).
"""

import asyncio
import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class OpenCodeBridge:
    """
    Puente para interactuar con OpenCode CLI.

    Soporta:
      - Ejecución headless de prompts (opencode run)
      - Gestión del workspace
      - Configuración de modelos y providers
      - Lectura de sesiones y logs
    """

    def __init__(
        self,
        binary: str = "opencode",
        workspace: str = "./output",
        provider: str = "ollama",
        model: str = "codellama:13b",
    ):
        self.binary = binary
        self.workspace = Path(workspace).resolve()
        self.provider = provider
        self.model = model
        self._initialized = False
        logger.info(f"OpenCodeBridge: workspace={self.workspace}")

    def _check_binary(self) -> bool:
        """Verificar que opencode está instalado."""
        return shutil.which(self.binary) is not None

    async def initialize(self) -> bool:
        """
        Inicializar el workspace de OpenCode.

        Returns:
            True si la inicialización fue exitosa
        """
        if not self._check_binary():
            logger.error(
                f"OpenCode CLI no encontrado. Instalar con: npm i -g opencode-ai"
            )
            return False

        # Crear workspace si no existe
        self.workspace.mkdir(parents=True, exist_ok=True)

        # Verificar si ya está inicializado
        config_file = self.workspace / ".opencode" / "config.json"
        if not config_file.exists():
            logger.info("Inicializando workspace de OpenCode...")
            result = await self._run_command("init", cwd=str(self.workspace))
            if result["returncode"] != 0:
                logger.warning(
                    "No se pudo inicializar automáticamente. "
                    "Ejecute 'opencode' en el directorio manualmente."
                )

        self._initialized = True
        return True

    async def _run_command(
        self,
        *args: str,
        cwd: Optional[str] = None,
        timeout: int = 300,
        env_extra: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Ejecutar un comando de OpenCode.

        Returns:
            Dict con stdout, stderr y returncode
        """
        cmd = [self.binary] + list(args)
        cwd = cwd or str(self.workspace)

        env = os.environ.copy()
        if env_extra:
            env.update(env_extra)

        logger.debug(f"Ejecutando: {' '.join(cmd)} en {cwd}")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            result = {
                "stdout": stdout.decode("utf-8", errors="replace"),
                "stderr": stderr.decode("utf-8", errors="replace"),
                "returncode": process.returncode,
            }

            if process.returncode != 0:
                logger.warning(f"Comando falló (code={process.returncode}): {stderr.decode()}")
            else:
                logger.debug(f"Comando exitoso: {stdout.decode()[:200]}")

            return result

        except asyncio.TimeoutError:
            logger.error(f"Timeout ejecutando: {' '.join(cmd)}")
            process.kill()
            return {"stdout": "", "stderr": "Timeout", "returncode": -1}
        except Exception as e:
            logger.error(f"Error ejecutando comando: {e}")
            return {"stdout": "", "stderr": str(e), "returncode": -1}

    async def run_prompt(
        self,
        prompt: str,
        project_dir: Optional[str] = None,
        mode: str = "build",
    ) -> Dict[str, Any]:
        """
        Ejecutar un prompt en modo headless con OpenCode.

        Args:
            prompt: El prompt/instrucción a ejecutar
            project_dir: Directorio del proyecto (default: workspace)
            mode: Modo de ejecución: "build" o "plan"

        Returns:
            Resultado con stdout, stderr y returncode
        """
        cwd = project_dir or str(self.workspace)
        Path(cwd).mkdir(parents=True, exist_ok=True)

        logger.info(f"OpenCode run ({mode}): {prompt[:100]}...")

        args = ["run", prompt]
        if mode == "plan":
            args.extend(["--agent", "plan"])

        return await self._run_command(*args, cwd=cwd)

    async def generate_file(
        self,
        filepath: str,
        description: str,
        project_dir: Optional[str] = None,
    ) -> bool:
        """
        Generar un archivo específico con OpenCode.

        Args:
            filepath: Ruta relativa del archivo a crear
            description: Descripción del contenido del archivo
            project_dir: Directorio del proyecto

        Returns:
            True si el archivo fue creado exitosamente
        """
        prompt = (
            f"Crea el archivo '{filepath}' con el siguiente contenido: {description}. "
            f"Asegúrate de que el código sea funcional, bien documentado, "
            f"y siga las mejores prácticas."
        )

        result = await self.run_prompt(prompt, project_dir=project_dir)

        if result["returncode"] == 0:
            target = Path(project_dir or self.workspace) / filepath
            if target.exists():
                logger.info(f"Archivo generado exitosamente: {filepath}")
                return True

        logger.warning(f"No se pudo verificar la creación de: {filepath}")
        return result["returncode"] == 0

    async def analyze_project(
        self, project_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analizar un proyecto existente con OpenCode en modo plan.

        Returns:
            Análisis del proyecto
        """
        prompt = (
            "Analiza este proyecto completo. Describe su estructura, "
            "tecnologías usadas, puntos de entrada, dependencias, "
            "y posibles mejoras. Responde en formato JSON estructurado."
        )

        result = await self.run_prompt(prompt, project_dir=project_dir, mode="plan")
        return result

    async def fix_issues(
        self,
        issues: List[str],
        project_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Corregir issues encontrados durante la revisión.

        Args:
            issues: Lista de problemas a corregir
            project_dir: Directorio del proyecto

        Returns:
            Resultado de las correcciones
        """
        issues_text = "\n".join(f"- {issue}" for issue in issues)
        prompt = (
            f"Corrige los siguientes problemas en el código:\n{issues_text}\n\n"
            f"Asegúrate de que las correcciones no rompan funcionalidad existente."
        )

        return await self.run_prompt(prompt, project_dir=project_dir)

    async def run_tests(self, project_dir: Optional[str] = None) -> Dict[str, Any]:
        """Ejecutar tests del proyecto."""
        prompt = (
            "Ejecuta todos los tests del proyecto y reporta los resultados. "
            "Si hay tests fallidos, describe las causas."
        )
        return await self.run_prompt(prompt, project_dir=project_dir)
