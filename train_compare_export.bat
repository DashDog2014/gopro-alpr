@echo off
call C:\Users\hcolt\anaconda3\Scripts\activate.bat alpr-clean
cd /d C:\Users\hcolt\gopro-alpr
python -c "import sys, torch, torchvision; print(sys.executable); print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python scripts\train_compare_export.py
pause