#!/bin/bash
cd $HOME
pkill -f rt_alphaface_server.py 2>/dev/null
sleep 1
setsid $HOME/codex_ffhq_realtime_pilot/venv/bin/python $HOME/rt_alphaface_server.py > $HOME/rt_alphaface.log 2>&1 < /dev/null &
exit 0
