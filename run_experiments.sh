#!/bin/bash
# 1. Train HKA
python -m src.hka_trainer --epochs 5

# 2. Train adapter for RAG
python -m src.adapter_tuner --epochs 3

# 3. Evaluate retrieval + generation
python -m src.rag_retriever --evaluate
