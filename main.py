from dotenv import load_dotenv
load_dotenv()

import os
from os import path
import json
from enum import Enum
from datetime import datetime
import base64
import re
import time
from typing import List, Tuple

from pyboy import PyBoy
import numpy as np
import openai

# PROMPS
DEFAULT_SYSTEM_PROMPT = (
    "You are an agent playing the game Pokemon Gold."
)

OBJECTIVE_SYSTEM_PROMPT = (
    "Your final goal is to finish the game.\n"
    "Current state of the game:\n"
    "{current_state}\n"
    "With that, you should decide on the next objective."
)

OBJECTIVE_USER_PROMPT = (
    "Skip the preamble, just say a brief sentence on the next immediate objective"
)

ACTION_SYSTEM_PROMPT = (
    "The user will send you screenshots of the game and you will have to tell them what button to press next. "
    "Reply in the following format:\n"
    "Thought: Your rationale for the next action\n"
    "Action: The next button to be pressed, without quotes. It should be one of: [A, B, UP, DOWN, LEFT, RIGHT, START, SELECT].\n"
    "The user will then reply you with the next screenshot.\n"
    "Once you finish your objective, say 'FINISH'.\n"
    "If the user tells you to summarize the state of the game, do that instead.\n\n"
    "Current state: {current_state}\n\n"
    "Current objective: {objective}"
)

SUMMARIZE_MESSAGE = (
    "Skip the preamble, summarize the current state of the game in details, including your current team, and progress on your current objective"
)

START_SUMARIZED_STATE = (
    "Just started the game. No pokemon caught."
)

# possible states:
# objective -> action -> summarize -> action -> summarize -> objective
class State(Enum):
    OBJECTIVE = 0
    ACTION = 1
    SUMMARIZE = 2

FRAME_SKIP = 60 # 60fps
STATE_JSON = 'state.json'
MODEL = 'gpt-4o'
MAX_TOKENS = 100
MAX_TOKENS_LONG = 512
MAX_ACTIONS = 10
IMAGE_FORMAT = 'png'
BASE_GAME_STATE_FILE = 'game_state_base.sav'
GAME_STATE_FILE = 'game_state.sav'

def compare_screenshots(prev_screenshot, curr_screenshot) -> float:
    # Compare the two screenshots
    _prev_np = np.array(prev_screenshot)
    _curr_np = np.array(curr_screenshot)
    total = np.ones(_prev_np.shape) * 255
    result = (_curr_np - _prev_np).sum()
    return 1 - result / total.sum()

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
        return f"data:image/png;base64,{base64_image}"

def save_state(state: dict):
    with open(STATE_JSON, 'w') as f:
        json.dump(state, f, indent=2)

def reset_history(state: dict):
    # archive history and prune current history
    if len(state['history']) > 0:
        state['full_history'].append(state['history'])
        state['history'] = []
    return state

def get_response(client: openai.Client, history: List[dict], max_tokens=MAX_TOKENS):
    response = client.chat.completions.create(
        model=MODEL,
        messages=history,
        max_tokens=max_tokens,
    )
    new_message = response.choices[0].message
    history.append(new_message.to_dict())
    return history, new_message

def get_next_objective(client: openai.Client, state: dict):
    state = reset_history(state)
    
    # get next objective
    objective_history = [
        {
            'role': 'system',
            'content': '\n'.join([
                DEFAULT_SYSTEM_PROMPT, 
                OBJECTIVE_SYSTEM_PROMPT.format(current_state=state['sumarized_state']),
            ]),
        },
        {
            'role': 'user',
            'content': OBJECTIVE_USER_PROMPT,
        }
    ]
    objective_history, new_message = get_response(client, objective_history)
    # archive the new history as well
    state['full_history'].append(objective_history)

    state['objective'] = new_message.content
    state['state'] = State.ACTION.value
    state['next_action'] = None

    save_state(state)
    return state

def get_next_action(client: openai.Client, state: dict, screenshot_url: str):
    if len(state['history']) == 0:
        state['history'] = [
            {
                'role': 'system',
                'content': '\n'.join([
                    DEFAULT_SYSTEM_PROMPT,
                    ACTION_SYSTEM_PROMPT.format(
                        objective=state['objective'],
                        current_state=state['sumarized_state'],
                    )
                ]),
            },
        ]
    
    state['history'].append({
        'role': 'user',
        'content': [
          {
            "type": "image_url",
            "image_url": {
              "url": screenshot_url,
            },
          },
        ],
    })
    start_time = time.time()
    state['history'], new_message = get_response(client, state['history'])

    if new_message.content == 'FINISH':
        state['state'] = State.OBJECTIVE.value
    else:
        regex = (
            r"Action\s*:(.*?)$"
        )
        action_match = re.search(regex, new_message.content, re.MULTILINE | re.DOTALL)
        if action_match:
            action = action_match.group(1).strip()
            state['current_actions'] += 1
            state['next_action'] = action.lower()
            print(f'Next action: {action} ({round(time.time() - start_time, 2)}s)')
            
            if state['current_actions'] >= MAX_ACTIONS:
                state['state'] = State.SUMMARIZE.value
                state['current_actions'] = 0

    save_state(state)
    return state

def get_summary(client: openai.Client, state: dict):
    state['history'].append({
        'role': 'user',
        'content': SUMMARIZE_MESSAGE,
    })
    state['history'], new_message = get_response(client, state['history'], max_tokens=MAX_TOKENS_LONG)
    state['sumarized_state'] = new_message.content

    # archive the new history as well to reset context length
    state['history'] = reset_history(state['history'])
    state['state'] = State.ACTION.value
    state['next_action'] = None

    save_state(state)
    return state

def main():
    BASE_URL = os.getenv('BASE_URL')
    if BASE_URL is None:
        client = openai.Client()
    else:
        client = openai.Client(base_url=BASE_URL)
    em = PyBoy('rom.gbc')
    
    i = 0
    prev_screenshot = None

    if path.exists(STATE_JSON):
        with open(STATE_JSON, 'r') as fp:
            state = json.load(fp)
    else:
        state = {
            'state_file': BASE_GAME_STATE_FILE,
            'history': [],
            'full_history': [],
            'objective': None,
            'sumarized_state': START_SUMARIZED_STATE,
            'state': State.OBJECTIVE.value,
            'current_actions': 0,
            'next_action': None,
        }

    if state['state_file'] != None:
        with open(state['state_file'], 'rb') as fp:
            em.load_state(fp)

    while em.tick():
        if i % FRAME_SKIP == 0:
            i = 0
            score = 0
            pil_image = em.screen.image
            pil_image.save('screenshot.png')

            if prev_screenshot != None:
                if score < 0.97 or state['next_action'] is None:
                    if state['state'] == State.ACTION.value:
                        if not path.isdir('./screenshots'):
                            os.mkdir('./screenshots')
                        image_path = f'./screenshots/{datetime.now().timestamp()}.{IMAGE_FORMAT}'
                        pil_image.save(image_path, optimize=True, quality=80)
                        image_url = encode_image(image_path)
                        state = get_next_action(client, state, image_url)

                        em.button(state['next_action'], 5)
                        em.tick()

                    elif state['state'] == State.SUMMARIZE.value:
                        # save new state after x turns
                        with open(GAME_STATE_FILE) as fp:
                            em.save_state(fp)
                        state['state_file'] = GAME_STATE_FILE

                        # reset model context
                        state = get_summary(client, state)

                    if state['state'] == State.OBJECTIVE.value:
                        state = get_next_objective(client, state)

                    score = compare_screenshots(prev_screenshot, pil_image)
                    prev_screenshot = pil_image
            else:
                prev_screenshot = pil_image
        i += 1

if __name__ == '__main__':
    main()