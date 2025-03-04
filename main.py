from dotenv import load_dotenv

load_dotenv()

import base64
import json
import os
import queue
import re
import threading
import time
from datetime import datetime
from enum import Enum
from os import path
from typing import List, Tuple

import numpy as np
import openai
from pyboy import PyBoy

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
    "Thought: Think of where you are and the next action you should do on the screen\n"
    "Action: The next button you should press on the screen, without quotes. It should be one of: [A, B, UP, DOWN, LEFT, RIGHT, START, SELECT].\n"
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
MAX_ACTIONS = 1
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
    # create new history with screenshot as last item only
    prompt_history = []
    for ix, item in enumerate(history):
        if not isinstance(item['content'], str) and ix != len(history) - 1:
            prompt_history.append({
                'role': item['role'],
                'content': '(image)',
            })
        else:
            prompt_history.append(item)
    
    response = client.chat.completions.create(
        model=MODEL,
        messages=prompt_history,
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
    
    screenshot_message = {
        'role': 'user',
        'content': [
          {
            "type": "image_url",
            "image_url": {
              "url": screenshot_url,
            },
          },
        ],
    }
    state['history'].append(screenshot_message)
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
    state = reset_history(state)
    state['state'] = State.ACTION.value
    state['next_action'] = None

    save_state(state)
    return state

# Communication queues between threads
action_queue = queue.Queue()  # For sending actions from API thread to PyBoy thread
screenshot_queue = queue.Queue()  # For sending screenshots from PyBoy thread to API thread
state_queue = queue.Queue()  # For sending state updates between threads

def pyboy_thread_function(state):
    """Thread function to run PyBoy and advance frames"""
    em = PyBoy('rom.gbc')
    i = 0
    prev_screenshot = None
    
    # Initialize state if it has a saved state file
    if state['state_file'] != None:
        with open(state['state_file'], 'rb') as fp:
            em.load_state(fp)
    
    # Put initial state in queue for API thread
    state_queue.put(state)
    
    while em.tick():
        # Check if there's a new action to perform
        try:
            action = action_queue.get_nowait()
            try:
                em.button(action, 5)
                em.tick()
            except Exception as e:
                print(f'Error: {e}, action: {action}')
            action_queue.task_done()
        except queue.Empty:
            # No new action, continue normal operation
            pass
            
        if i % FRAME_SKIP == 0:
            i = 0
            pil_image = em.screen.image
            pil_image.save('screenshot.png')
            
            if prev_screenshot != None:
                score = compare_screenshots(prev_screenshot, pil_image)
                if score < 0.97:  # Screen has changed significantly
                    # Save screenshot for API thread
                    if not path.isdir('./screenshots'):
                        os.mkdir('./screenshots')
                    image_path = f'./screenshots/{datetime.now().timestamp()}.{IMAGE_FORMAT}'
                    pil_image.save(image_path, optimize=True, quality=80)
                    image_url = encode_image(image_path)
                    
                    # Put screenshot in queue for API thread
                    if screenshot_queue.qsize() == 0:
                        screenshot_queue.put(image_url)
                    
                    # Save state periodically
                    if state['state'] == State.SUMMARIZE.value:
                        with open(GAME_STATE_FILE, 'wb') as fp:
                            em.save_state(fp)
                        state['state_file'] = GAME_STATE_FILE
                        
            prev_screenshot = pil_image
        
        i += 1
        
        # Update state from API thread if available
        try:
            new_state = state_queue.get_nowait()
            state = new_state
            state_queue.task_done()
        except queue.Empty:
            # No state update, continue
            pass

def api_thread_function(client):
    """Thread function to handle API requests"""
    # Get initial state from PyBoy thread
    state = state_queue.get()
    state_queue.task_done()
    
    while True:
        if state['state'] == State.OBJECTIVE.value:
            # Get next objective
            state = get_next_objective(client, state)
            # Share updated state with PyBoy thread
            state_queue.put(state)
            
        elif state['state'] == State.ACTION.value:
            # Wait for a new screenshot
            image_url = screenshot_queue.get()
            screenshot_queue.task_done()
            
            # Get next action
            if state['next_action'] is None:
                state = get_next_action(client, state, image_url)
                
                # If we have an action, send it to PyBoy thread
                if state['next_action'] is not None:
                    action_queue.put(state['next_action'])
                    
                # Share updated state with PyBoy thread
                state_queue.put(state)
                
        elif state['state'] == State.SUMMARIZE.value:
            # Reset model context
            state = get_summary(client, state)
            # Share updated state with PyBoy thread
            state_queue.put(state)
            
        # Small sleep to prevent CPU hogging
        time.sleep(0.01)

def main():
    global MAX_ACTIONS
    BASE_URL = os.getenv('BASE_URL')
    if BASE_URL is None:
        client = openai.Client()
        MAX_ACTIONS = 5
    else:
        print('Using BASE_URL:', BASE_URL)
        client = openai.Client(base_url=BASE_URL)
        MAX_ACTIONS = 3
    
    # Initialize state
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
    
    # Create and start PyBoy thread
    pyboy_thread = threading.Thread(target=pyboy_thread_function, args=(state,))
    pyboy_thread.daemon = True  # Thread will exit when main program exits
    pyboy_thread.start()
    
    # Create and start API thread
    api_thread = threading.Thread(target=api_thread_function, args=(client,))
    api_thread.daemon = True  # Thread will exit when main program exits
    api_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == '__main__':
    main()