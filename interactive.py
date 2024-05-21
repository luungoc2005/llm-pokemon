from pyboy import PyBoy
import time

STATE_FILE = 'game_state.sav'

em = PyBoy('rom.gbc')
start_time = time.time()
while em.tick():
    # if 30s passed:
    if time.time() - start_time >= 60 * 5:
        with open(STATE_FILE, 'wb') as fp:
            em.save_state(fp)
            em.stop()
            exit()