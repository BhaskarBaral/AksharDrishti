"""
Run one PaddleX training/evaluate/export config. Thin wrapper around
paddlex.engine.Engine, which is normally driven by a `main.py` console
entrypoint that this pip install doesn't expose as a script -- calling it
directly avoids depending on that.

Usage:
  .venv\\Scripts\\python.exe training/train_one.py -c training/configs/devanagari.yaml
  .venv\\Scripts\\python.exe training/train_one.py -c training/configs/devanagari.yaml -o Global.mode=evaluate
"""
from paddlex.engine import Engine

if __name__ == "__main__":
    Engine().run()
