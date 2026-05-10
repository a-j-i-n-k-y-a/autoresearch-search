## Quick start
pip install -r requirements.txt
python prepare.py          # one-time data setup
python agent_loop.py --eval-only   # benchmark current search.py
python agent_loop.py --n 20 --objective recall   # run experiments