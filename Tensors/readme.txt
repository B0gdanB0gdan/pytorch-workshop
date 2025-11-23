1. Create a virtual environment (venv or conda)
2. Install dependencies
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install notebook matplotlib pandas Pillow
3. Move attached assets folder on the same level with the notebook for image rendering inside markup cells
4. Install the env to be used in Jupyter:
python -m ipykernel install --user --name <your_env_name>


Option2: Colab