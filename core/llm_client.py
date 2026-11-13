"""
core/llm_client.py - Cliente para comunicación con Ollama.

Gestiona la conexión con el servidor Ollama local y proporciona
una interfaz unificada para generación de texto y código.
"""

import asyncio
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)


class OllamaClient:
    """
    Cliente asíncrono para el servidor Ollama.

    Proporciona:
      - Generación de texto/código
      - Chat con historial
      - Streaming de respuestas
      - Manejo de múltiples modelos
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        default_model: str = "codellama:13b",
        timeout: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.default_model = default_model
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        logger.info(f"OllamaClient configurado: {base_url} / modelo: {default_model}")

    async def _get_client(self) -> httpx.AsyncClient:
        """Obtener o crear el cliente HTTP."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout, connect=10.0),
            )
        return self._client

    async def close(self) -> None:
        """Cerrar la conexión."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def is_available(self) -> bool:
        """Verificar si Ollama está disponible."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Ollama no disponible: {e}")
            return False

    async def list_models(self) -> List[str]:
        """Listar modelos disponibles en Ollama."""
        try:
            client = await self._get_client()
            response = await client.get("/api/tags")
            response.raise_for_status()
            data = response.json()
            return [m["name"] for m in data.get("models", [])]
        except Exception as e:
            logger.error(f"Error listando modelos: {e}")
            return []

    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Generar texto/código con Ollama.

        Args:
            prompt: El prompt del usuario
            model: Modelo a usar (por defecto el configurado)
            system: Prompt de sistema
            temperature: Temperatura de generación
            max_tokens: Máximo de tokens a generar

        Returns:
            Texto generado
        """
        model = model or self.default_model
        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        logger.debug(f"Generando con {model}, tokens={max_tokens}")

        try:
            response = await client.post("/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
            result = data.get("response", "")
            logger.info(
                f"Generación completada: {len(result)} chars, "
                f"eval_duration={data.get('eval_duration', 'N/A')}"
            )
            return result
        except httpx.TimeoutException:
            logger.error("Timeout al generar con Ollama")
            raise
        except Exception as e:
            logger.error(f"Error generando: {e}")
            raise

    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 4096,
    ) -> str:
        """
        Chat con historial de mensajes.

        Args:
            messages: Lista de mensajes [{"role": "user/assistant/system", "content": "..."}]
            model: Modelo a usar
            temperature: Temperatura
            max_tokens: Máximo de tokens

        Returns:
            Respuesta del asistente
        """
        model = model or self.default_model
        client = await self._get_client()

        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            response = await client.post("/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            return data.get("message", {}).get("content", "")
        except Exception as e:
            logger.error(f"Error en chat: {e}")
            raise

    async def generate_stream(
        self,
        prompt: str,
        model: Optional[str] = None,
        system: Optional[str] = None,
        temperature: float = 0.3,
    ) -> AsyncGenerator[str, None]:
        """
        Generar texto con streaming.

        Yields:
            Fragmentos de texto generado
        """
        model = model or self.default_model
        client = await self._get_client()

        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": temperature},
        }
        if system:
            payload["system"] = system

        try:
            async with client.stream(
                "POST", "/api/generate", json=payload
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Error en streaming: {e}")
            raise

    async def generate_code(
        self,
        description: str,
        language: str = "python",
        context: str = "",
        model: Optional[str] = None,
    ) -> str:
        """
        Generar código a partir de una descripción.

        Args:
            description: Qué debe hacer el código
            language: Lenguaje de programación
            context: Contexto adicional (código existente, etc.)
            model: Modelo a usar

        Returns:
            Código generado
        """
        system_prompt = f"""Eres un experto programador en {language}.
Tu tarea es generar código limpio, bien documentado y funcional.
REGLAS:
- Devuelve SOLO el código, sin explicaciones adicionales
- Incluye docstrings y comentarios relevantes
- Sigue las mejores prácticas del lenguaje
- Maneja errores apropiadamente
- El código debe ser ejecutable directamente"""

        prompt = f"Genera código en {language} para: {description}"
        if context:
            prompt += f"\n\nContexto adicional:\n{context}"

        return await self.generate(
            prompt=prompt,
            system=system_prompt,
            model=model,
            temperature=0.2,
        )
