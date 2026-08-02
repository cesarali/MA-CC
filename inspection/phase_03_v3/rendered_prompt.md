# Compiled prompt

- Prompt: `basic_choice@1`
- Definition hash: `e7196f7ea821ca3823211ae87a9268b3004730dfce0bdcc25f8a13a1801f7625`
- Instance hash: `3e7f63fb48e977392ba66097e53fc6d9d272256f24933d58f9c3a72efad941b0`
- Tokenizer: `mas_cc_regex_v1_estimate`
- Block tokens: `133`

## 1. Task

Role: `system`  
Block: `task@1`  
Tokens: `10`

Choose the option that best matches your private signal.

## 2. Rules

Role: `system`  
Block: `rules@1`  
Tokens: `19`

["Choose exactly one available option.", "Do not reveal hidden metadata."]

## 3. Private state

Role: `system`  
Block: `private_state@1`  
Tokens: `27`

{"available_actions": ["A", "B"], "committed_action": null, "cumulative_score": 50}

## 4. Recent memory

Role: `system`  
Block: `recent_memory@1`  
Tokens: `50`

[{"other_action": "B", "own_action": "A", "payoff": -50}, {"other_action": "B", "own_action": "B", "payoff": 100}]

## 5. Current interaction

Role: `system`  
Block: `current_interaction@1`  
Tokens: `27`

{"available_actions": ["A", "B"], "interaction_number": 3, "other_action_visible": false}
