# Adaptive_Geo_Knowledge_Extraction

Starting to work on refatcoring...
tmux test

## Overview
An automated multi-agent system that leverages Large Language Models (LLMs) to extract and standardize knowledge from unstructured Earth Science documents into structured knowledge bases, with minimal human intervention.

## Key Features
- Automated knowledge extraction from geological documents
- LLM-powered multi-agent architecture
- Adaptable to different controlled vocabularies
- Modularized framework design

## Getting Started

The entry point for this project is `workflow_main.py`. The LLM API key is not included in the repo. To run this project, please configure the LLM service parameters in your environment.
This project uses the Azure OpenAI Service for development, which requires additional parameters such as the development name. If you are using another LLM service, you will need to import different LLM supporting packages.
