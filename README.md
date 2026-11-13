# 🤖 OpenCode Agentic System

Sistema de inteligencia artificial agéntica con **3 agentes** y **4 skills** para la generación autónoma de programas y código fuente utilizando **OpenCode** y **Ollama**.

---

## 🏛️ Arquitectura

- **3 Agentes:**
  - `PlannerAgent`: Arquitecto y planificador de tareas y módulos.
  - `BuilderAgent`: Constructor y generador de código fuente.
  - `ReviewerAgent`: Revisor de calidad, bugs y mejores prácticas.

- **4 Skills:**
  - `RequirementsAnalysisSkill`: Análisis de requisitos y especificaciones.
  - `CodeGenerationSkill`: Generación de código multi-lenguaje.
  - `AutoTestingSkill`: Generación de pruebas unitarias y cálculo de cobertura.
  - `DocumentationSkill`: Creación de documentación, diagramas Mermaid y guías.

---

## 🚀 Instalación y Uso

```bash
# 1. Crear y activar entorno virtual (opcional pero recomendado)
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate # Linux/Mac

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
python main.py "Crear una API REST para gestión de inventario con FastAPI"

# O en modo interactivo
python main.py --interactive
```
