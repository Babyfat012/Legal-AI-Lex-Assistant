# Introduction

Legal AI - Lex Assistant is a cloud-native Law Question Answering system built with Retrieval-Agumented Generation (RAG) architecture. The system's UI is built by Streamlit, and FastAPI, a vector search for knowledge retrieval, and an inference server, with Redis as a cached chat history for multi-run conversations. This system is deployed on Kubernetes, automated by Terraform, Helm and Jenkins CI/CD, fully observed and monitored using Prometheus, Grafana, ELK stacks (Elasicsearch, Logstack, Kibana) and tracing with Jaeger.

This repository is aim to learning an end-to-end RAG workflow, suitable for people who need search quickly  laws questions with a correct answer.

# Repository structure
