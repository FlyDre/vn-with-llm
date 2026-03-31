# create .venv folder

py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

# run install.ps

.\install.ps1

<br />

llm生成信号：\
.\.venv\Scripts\python.exe .\scripts\generate_llm_signals.py --vt-symbol 000559.SZSE --start 2025-12-01 --end 2026-03-27 --window 20 --capital 400000 --lot-size 100 --output .\data\llm_signals\000559.SZSE_20240101_20250630.csv

.\.venv\Scripts\python.exe .\scripts\generate_llm_signals.py --vt-symbol 603081.SSE --start 2025-10-21 --end 2026-03-28 --window 20 --capital 400000 --lot-size 100 --output .\data\llm_signals\603081.csv --strict-llm

.\.venv\Scripts\python.exe .\examples\veighna_trader\run.py