# Compiled prompt

- Prompt: `basic_binary_choice@1`
- Tokenizer: `mas_cc_regex_v1_estimate`
- Total tokens: `200`

## 1. Task description

Role: `system`  
Block: `task_description@1`  
Tokens: `13`

Coordinate with another player by choosing one of the two available actions.

## 2. Game rules

Role: `system`  
Block: `game_rules@1`  
Tokens: `52`

1. Choose exactly one action on every interaction.
2. Both players receive a positive payoff when their actions match.
3. Both players receive a negative payoff when their actions differ.
4. The other player's current choice is not visible before you decide.

## 3. Private information

Role: `user`  
Block: `private_state@1`  
Tokens: `23`

- Available actions: ["A", "B"]
- Cumulative score: 50
- Committed action: null

## 4. Recent memory

Role: `user`  
Block: `recent_memory@1`  
Tokens: `55`

- Interaction 1: {"other_action": "B", "own_action": "A", "payoff": -50}
- Interaction 2: {"other_action": "B", "own_action": "B", "payoff": 100}

## 5. Current interaction

Role: `user`  
Block: `current_interaction@1`  
Tokens: `24`

- Interaction number: 3
- Available actions: ["A", "B"]
- Other action visible: false

## 6. Decision instruction

Role: `user`  
Block: `decision_instruction@1`  
Tokens: `11`

Select the action that you will play in this interaction.

## 7. Output contract

Role: `user`  
Block: `output_contract@1`  
Tokens: `22`

Return exactly one of these values: A, B. Return only the value, with no explanation or punctuation.
