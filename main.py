#!/usr/bin/env python3
"""
main.py - Punto de entrada del OpenCode Agentic System.

CLI para generar proyectos completos utilizando un sistema de IA agéntica
con 3 agentes y 4 skills, integrado con OpenCode y Ollama.

Uso:
    python main.py "Crear una API REST para gestión de tareas"
    python main.py --language javascript "Crear un chat en tiempo real"
    python main.py --interactive
"""

import asyncio
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from core.config import load_config
from core.orchestrator import Orchestrator

console = Console()

BANNER = r"""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║     ___                   ____          _                    ║
║    / _ \ _ __   ___ _ __ / ___|___   __| | ___              ║
║   | | | | '_ \ / _ \ '_ | |   / _ \ / _` |/ _ \            ║
║   | |_| | |_) |  __/ | || |__| (_) | (_| |  __/            ║
║    \___/| .__/ \___|_| |_\____\___/ \__,_|\___|             ║
║         |_|                                                  ║
║                                                              ║
║   🤖  A G E N T I C   S Y S T E M   v1.0                   ║
║                                                              ║
║   Agentes: Planificador │ Constructor │ Revisor              ║
║   Skills:  Requisitos │ Código │ Tests │ Docs                ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""


@click.command()
@click.argument("description", required=False)
@click.option(
    "--language", "-l",
    default="python",
    help="Lenguaje de programación (python, javascript, typescript, go, etc.)",
)
@click.option(
    "--output", "-o",
    default=None,
    help="Directorio de salida para el proyecto generado",
)
@click.option(
    "--config", "-c",
    default=None,
    help="Ruta al archivo de configuración YAML",
)
@click.option(
    "--interactive", "-i",
    is_flag=True,
    help="Modo interactivo con prompts paso a paso",
)
@click.option(
    "--model",
    default=None,
    help="Modelo de Ollama a usar (ej: codellama:13b, llama3:8b)",
)
def main(
    description: str,
    language: str,
    output: str,
    config: str,
    interactive: bool,
    model: str,
):
    """
    🤖 OpenCode Agentic System - Genera proyectos completos con IA.

    Proporciona una DESCRIPTION del proyecto que quieres crear,
    o usa --interactive para el modo guiado.

    Ejemplos:

      python main.py "Crear una calculadora científica con GUI"

      python main.py -l javascript "Crear un servidor HTTP con autenticación"

      python main.py --interactive
    """
    console.print(BANNER, style="bold cyan")

    # Modo interactivo
    if interactive or not description:
        description = _interactive_mode(language)
        if not description:
            console.print("[red]No se proporcionó descripción. Saliendo.[/]")
            sys.exit(1)

    # Cargar configuración
    cfg = load_config(config)

    # Sobrescribir modelo si se especificó
    if model:
        cfg.llm.model = model

    # Crear orquestador y ejecutar
    orchestrator = Orchestrator(config=cfg)

    try:
        result = asyncio.run(
            orchestrator.create_project(
                description=description,
                language=language,
                output_dir=output,
            )
        )

        if result.get("success"):
            console.print(
                Panel(
                    f"[bold green]🎉 ¡Proyecto generado exitosamente![/]\n\n"
                    f"Directorio: [cyan]{result['project_dir']}[/]\n"
                    f"Archivos: [cyan]{result['files_count']}[/]\n"
                    f"Puntuación: [cyan]{result['review_score']}/10[/]",
                    title="Resultado",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]Error:[/] {result.get('error', 'Unknown')}",
                    title="Error",
                    border_style="red",
                )
            )
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Operación cancelada por el usuario.[/]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Error fatal: {e}[/]")
        logger_main = __import__("logging").getLogger(__name__)
        logger_main.error(f"Error fatal: {e}", exc_info=True)
        sys.exit(1)


def _interactive_mode(default_language: str) -> str:
    """Modo interactivo para recopilar información del proyecto."""
    console.print(
        Panel(
            "[bold]Modo Interactivo[/]\n"
            "Responde las siguientes preguntas para configurar tu proyecto.",
            border_style="yellow",
        )
    )

    # Descripción del proyecto
    console.print("\n[bold cyan]¿Qué proyecto quieres crear?[/]")
    console.print("[dim]Describe el proyecto en lenguaje natural. Sé lo más específico posible.[/]")
    description = Prompt.ask("\n💡 Descripción")

    if not description.strip():
        return ""

    # Detalles adicionales
    console.print("\n[bold cyan]¿Algún detalle adicional?[/] [dim](presiona Enter para omitir)[/]")
    extra = Prompt.ask("📝 Detalles", default="")

    if extra.strip():
        description += f". Detalles adicionales: {extra}"

    # Confirmación
    console.print(
        Panel(
            f"[white]{description}[/]",
            title="Tu proyecto",
            border_style="cyan",
        )
    )

    confirm = Prompt.ask(
        "\n¿Iniciar generación?", choices=["si", "no"], default="si"
    )

    if confirm.lower() != "si":
        return ""

    return description


if __name__ == "__main__":
    main()
