# LLM-Integrated FMEA System

A comprehensive multi-agent system for intelligent Failure Mode and Effects Analysis (FMEA) generation using document extraction, natural language processing, and retrieval-augmented generation (RAG).

## Features

- **Multi-Agent Pipeline**: Automated component and subsystem extraction from technical documents
- **Intelligent FMEA Generation**: Historical data-driven risk assessment with detailed recommendations
- **RAG-Based Document Chat**: Interactive Q&A with uploaded technical documentation
- **Visual Analytics**: Comprehensive dashboards for risk analysis and component visualization
- **Local & Cloud LLM Support**: Compatible with Ollama (local) models

## Architecture

The system is built using a modular agent-based architecture:

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Coordinator                        │
│                 (Main Orchestrator)                         │
└─────────────────────────┬───────────────────────────────────┘
                          │
    ┌─────────────────────┼─────────────────────┐
    │                     │                     │
    v                     v                     v
┌─────────┐       ┌──────────────┐      ┌─────────────┐
│  Chat   │       │   Visual     │      │    FMEA     │
│ Agent   │       │  Dashboard   │      │ Generator   │
└─────────┘       └──────────────┘      └─────────────┘
    │                     │                     │
    └─────────────────────┼─────────────────────┘
                          │
                          v
                ┌─────────────────┐
                │ Document Agent  │
                │ (Core Engine)   │
                └─────────────────┘
```

## Components

### Core Files

- **`document_agent.py`**: Core extraction and RAG engine with LlamaIndex integration
- **`agent_coordinator.py`**: Main Streamlit application orchestrator
- **`chat_agent.py`**: Document chat interface with RAG capabilities
- **`visual_dashboard.py`**: Visualization and analytics engine
- **`fmea_generator_agent.py`**: FMEA analysis and risk assessment generator

## Installation

### Prerequisites

- Python
- Ollama (for local LLM support) 

## Quick Start

1. **Launch the application**
```bash
cd agents
streamlit run agent_coordinator.py
```

2. **Initialize the system**
   - Configure LLM settings in the sidebar
   - Click "🚀 Initialize Agent"

3. **Upload documents**
   - Go to "📤 Upload & Process" tab
   - Upload PDF, DOCX, XLSX, CSV, or TXT files
   - Include historical maintenance data for enhanced FMEA

4. **Run extraction pipeline**
   - Navigate to "🤖 Multi-Agent Pipeline" tab
   - Click "🚀 Run Pipeline"
   - Review extracted components and subsystems

5. **Generate FMEA**
   - Go to "🎯 FMEA Analysis" tab
   - Click "🎯 Generate FMEA"
   - Analyze risk assessments and recommendations

6. **Chat with documents**
   - Visit "💬 Chat with Documents" tab
   - Build search indexes
   - Ask questions about your documentation

## Supported Document Types

| Format | Purpose | Features |
|--------|---------|----------|
| **PDF** | Technical manuals, maintenance logs | Text extraction, OCR-ready |
| **DOCX** | Reports, procedures | Full text and table extraction |
| **XLSX/CSV** | Historical data, component lists | Structured data processing |
| **TXT** | Plain text documentation | Direct text processing |

## Configuration Options

### LLM Settings
- **Local LLM**: Ollama with models like llama3.2:1b, gemma:2b
- **Embedding Models**: HuggingFace models for semantic search

### Pipeline Parameters
- **Top-k Candidates**: Number of terms to extract (50-1000)
- **Similarity Threshold**: Term merging sensitivity (0.6-0.95)
- **Classification**: Heuristic or LLM-based term classification

## FMEA Intelligence Features

### Historical Data Integration
- Automatic severity extraction from maintenance logs
- Occurrence calculation based on failure frequency
- Intelligent failure mode prediction

### Risk Assessment
- **Severity**: 1-10 scale with intelligent keyword analysis
- **Occurrence**: Historical frequency-based calculations  
- **Detection**: Component accessibility and monitoring assessment
- **RPN**: Risk Priority Number (Severity × Occurrence × Detection)

### Recommendations Engine
- Component-specific maintenance procedures
- Condition monitoring strategies
- Training and redundancy recommendations
- Inspection frequency optimization

## Output Formats

### FMEA Table Columns
- Component, Potential Failure Mode, Potential Causes, Potential Effects
- Severity (S), Occurrence (O), Detection (D), RPN
- Severity Level, Reasoning for each rating
- Detailed Recommended Actions, Data Source

### Export Options
- **CSV**: Spreadsheet-compatible FMEA tables
- **JSON**: Structured data for integration
- **High Risk Items**: Filtered exports for critical components

## Performance Optimization

### Memory Usage
- Document chunking for large files
- Incremental processing for datasets
- Configurable embedding dimensions

### Processing Speed
- Parallel candidate extraction
- Cached similarity calculations
- Optimized vector operations

## Troubleshooting

### Common Issues

**Ollama Connection Failed**
```bash
# Ensure Ollama is running
ollama serve
# Check model availability
ollama list
```

**Memory Errors**
- Reduce chunk size in configuration
- Process fewer documents simultaneously
- Use smaller embedding models

**Poor Extraction Quality**
- Include domain-specific seed terms
- Adjust similarity thresholds
- Use LLM classification for better accuracy

### Debug Mode
Enable debug information in the sidebar to view:
- Session state variables
- Pipeline execution logs
- FMEA generation metadata

