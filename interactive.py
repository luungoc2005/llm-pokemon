from pyboy import PyBoy
from os import path
import time

STATE_FILE = 'game_state.sav'

em = PyBoy('rom.gbc')
if path.exists(STATE_FILE):
    with open(STATE_FILE, 'rb') as fp:
        em.load_state(fp)
start_time = time.time()
while em.tick():
    # if 30s passed:
    if time.time() - start_time >= 60 * 5:
        with open(STATE_FILE, 'wb') as fp:
            em.save_state(fp)
            em.stop()
            exit()