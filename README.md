# Wingman-Chatterbox-Openai-Server
Openai-compatible Text to Speech Server for Chatterbox TTS

## Installation
1. Tested with python 3.11.7
2. Download the code of this repository (https://github.com/teddybear082/Wingman-Chatterbox-Openai-Server/archive/refs/heads/main.zip)
3. create a virtual environment called venv
4. pip install either cpu_requirements.txt (should work for cpu pr apple mps generations) or cuda_requirements.txt (should work for nvidia gpus)
5. Run python wingman_chatterbox_openai_server.py followed by any command line arguments (can see arguments with --help), or run run_server_with_python.bat on windows
6. A simple web page to test generations will be available at http://localhost:5002 after server is running, and openai compatible endpoint will be available at http://localhost:5002/v1
