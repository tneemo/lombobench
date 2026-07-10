"""LomboBench — observabilidade caixa-preta dos controles externos de inferência do Ollama.

Fase 1 do LomboAI: proxy transparente + registro em SQLite + Lab de experimentação.
Este pacote NÃO acessa pesos/ativações do modelo; ele observa parâmetros de
amostragem, contexto e comportamento (métricas, texto, thinking).
"""

__version__ = "0.1.0"
