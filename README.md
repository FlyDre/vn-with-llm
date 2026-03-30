# create .venv folder

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# run install.ps

.\install.ps1

<br />

llm生成信号：\
.\.venv\Scripts\python.exe .\scripts\generate\_llm\_signals.py --vt-symbol 000559.SZSE --start 2025-12-01 --end 2026-03-27 --window 20 --capital 400000 --lot-size 100 --output .\data\llm\_signals\000559.SZSE\_20240101\_20250630.csv
