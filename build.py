# -*- coding: utf-8 -*-
import sys
from PyInstaller.__main__ import run

if __name__ == "__main__":
    print("Building MppleAI.exe...")
    
    args = [
        "app.py",
        "--name=MppleAI",
        "--onefile",
        "--windowed",
        "--icon=NONE",
        "--add-data=chat_model.pkl;.",
        "--hidden-import=torch",
        "--hidden-import=numpy",
        "--hidden-import=tkinter",
        "--collect-all=torch",
        "--collect-all=numpy",
    ]
    
    run(args)
