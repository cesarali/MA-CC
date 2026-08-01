# Paper-grounded prompt examples

Each section shows one complete request exactly as it would be passed to an LLM provider.

---

# Social conventions paper — one agent decision

- Prompt version: `social_conventions_paper@1`
- Messages sent: `2`
- Token counter: `mas_cc_regex_v1_estimate`
- Estimated block tokens: `310`

## Request metadata

```json
{
  "fixture": "social_conventions_supplement_example_v1",
  "source": "pdfs/Emergence of social conventions supplementary.pdf"
}
```

## Exact messages sent to the LLM

The messages below are shown in transmission order. Text inside each fence is the exact message content.

### Message 1 — `system`

```text
Context: Player 1 is playing a multi-round partnership game with Player 2 for 100 rounds. At each round, Player 1 and Player 2 simultaneously pick an action from the following values: [F, J]. The payoff that both players get is determined by the following rule:

1. If Players play the SAME action as each other, they will both be REWARDED with payoff 100 points.
2. If Players play DIFFERENT actions to each other, they will both be PUNISHED with payoff -50 points.

The objective of each Player is to maximize their own accumulated point tally, conditional on the behavior of the other player.
This is the history of choices in past rounds:
{'round': 1, 'Player 1': 'F', 'Other Player': 'J', 'payoff': -50}
{'round': 2, 'Player 1': 'J', 'Other Player': 'J', 'payoff': 100}
{'round': 3, 'Player 1': 'J', 'Other Player': 'J', 'payoff': 100}

It is now round 4. The current score of Player 1 is 150. Answer saying which value Player 1 should pick. Please think step by step before making a decision. Remember, examining history explicitly is important.

Write your answer using the following answer-first format: {'value': <F or J>; 'reason': <YOUR REASON>}.
```

### Message 2 — `user`

```text
Answer saying which action Player 1 should play.
```

## Response contract

```json
{
  "allowed_values": [
    "F",
    "J"
  ],
  "type": "paper_choice_reason"
}
```

## Block provenance

| Order | Block | Version | Role | Estimated tokens |
|---:|---|---:|---|---:|
| 1 | `partnership_context` | 1 | `system` | 56 |
| 2 | `payoff_rules` | 1 | `system` | 44 |
| 3 | `bounded_memory` | 1 | `system` | 127 |
| 4 | `round_state` | 1 | `system` | 42 |
| 5 | `output_contract` | 1 | `system` | 32 |
| 6 | `decision_request` | 1 | `user` | 9 |


---

# HiddenBench paper — first public speaker

- Prompt version: `hidden_profile_discussion_paper@1`
- Messages sent: `2`
- Token counter: `mas_cc_regex_v1_estimate`
- Estimated block tokens: `466`

## Request metadata

```json
{
  "agent_id": 0,
  "audit_answer_included": false,
  "fixture": "hiddenbench_downloaded_task_v1",
  "shuffle_seed": 1026,
  "source_data": "/home/cesarali/LanguageGames/MA-CC/scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/scaled/exact_replication/N_32.json",
  "task_id": 1,
  "transcript_is_inspection_fixture": true
}
```

## Exact messages sent to the LLM

The messages below are shown in transmission order. Text inside each fence is the exact message content.

### Message 1 — `system`

```text
You are participating in a study, acting as a community leader of a small village surrounded by mountains and rivers. Most villagers own cars, but there are also elderly people and children who may need additional assistance when walking. Earlier today, heavy rain began to fall, and the local government issued a warning about a potential disaster.
Hours ago, you requested relief supplies, but the supply truck has yet to arrive. Now, the rain has temporarily stopped, giving you and the others community leaders a short window to decide on the safest evacuation route before the rain resumes. You don’t know how much time you have left to make this critical decision.
Your Task:
you will discuss with other participants, who are also acting as community leaders, to decide where to evacuate. You have three options:
- West City: Accessible through a bridge over the river.
- East Town: Accessible through a tunnel on middle ground.
- North Hill: Accessible through a driveway and walking trails.
Usually, it takes the same time to reach all three places by car, but some routes may be inaccessible now.
There is only one correct evacuation location. After the discussion:
- If you choose the correct location, you will earn $1.
- If all other participants also choose the correct location, you will earn an additional $1 (for a total of $2).
This means that coordinating with others is critical to maximize your rewards. The chat will at most take 15 minutes. However, the exact time when the chat will end is unknown.

You have received the following information. Notice that the order of this information is randomly shuffled; the order of facts does not indicate importance or relationship. Please reason carefully:
- A massive fire has blocked the supply truck and all other traffic.
- The school at North Hill can serve as a temporary evacuation center, providing a two-week supply of essentials and sleeping space in the gym.
- The mayor of East Town has offered accommodations for any evacuees. She also ensures that volunteers are available to assist them.
- The river level is still below the bridge to West City.
- The local government announced that hotels in West City are prepared to accommodate evacuees. While these hotels are fully stocked with food, they may lack medical supplies.

Keep your response concise—just one or two sentences.
```

### Message 2 — `user`

```text
You are the first to speak.
```

## Response contract

```json
{
  "type": "free_text"
}
```

## Block provenance

| Order | Block | Version | Role | Estimated tokens |
|---:|---|---:|---|---:|
| 1 | `scenario_description` | 1 | `system` | 304 |
| 2 | `available_information` | 1 | `system` | 144 |
| 3 | `brevity_instruction` | 1 | `system` | 11 |
| 4 | `discussion_turn` | 1 | `user` | 7 |


---

# HiddenBench paper — one public discussion turn

- Prompt version: `hidden_profile_discussion_paper@1`
- Messages sent: `2`
- Token counter: `mas_cc_regex_v1_estimate`
- Estimated block tokens: `515`

## Request metadata

```json
{
  "agent_id": 0,
  "audit_answer_included": false,
  "fixture": "hiddenbench_downloaded_task_v1",
  "shuffle_seed": 1026,
  "source_data": "/home/cesarali/LanguageGames/MA-CC/scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/scaled/exact_replication/N_32.json",
  "task_id": 1,
  "transcript_is_inspection_fixture": true
}
```

## Exact messages sent to the LLM

The messages below are shown in transmission order. Text inside each fence is the exact message content.

### Message 1 — `system`

```text
You are participating in a study, acting as a community leader of a small village surrounded by mountains and rivers. Most villagers own cars, but there are also elderly people and children who may need additional assistance when walking. Earlier today, heavy rain began to fall, and the local government issued a warning about a potential disaster.
Hours ago, you requested relief supplies, but the supply truck has yet to arrive. Now, the rain has temporarily stopped, giving you and the others community leaders a short window to decide on the safest evacuation route before the rain resumes. You don’t know how much time you have left to make this critical decision.
Your Task:
you will discuss with other participants, who are also acting as community leaders, to decide where to evacuate. You have three options:
- West City: Accessible through a bridge over the river.
- East Town: Accessible through a tunnel on middle ground.
- North Hill: Accessible through a driveway and walking trails.
Usually, it takes the same time to reach all three places by car, but some routes may be inaccessible now.
There is only one correct evacuation location. After the discussion:
- If you choose the correct location, you will earn $1.
- If all other participants also choose the correct location, you will earn an additional $1 (for a total of $2).
This means that coordinating with others is critical to maximize your rewards. The chat will at most take 15 minutes. However, the exact time when the chat will end is unknown.

You have received the following information. Notice that the order of this information is randomly shuffled; the order of facts does not indicate importance or relationship. Please reason carefully:
- A massive fire has blocked the supply truck and all other traffic.
- The school at North Hill can serve as a temporary evacuation center, providing a two-week supply of essentials and sleeping space in the gym.
- The mayor of East Town has offered accommodations for any evacuees. She also ensures that volunteers are available to assist them.
- The river level is still below the bridge to West City.
- The local government announced that hotels in West City are prepared to accommodate evacuees. While these hotels are fully stocked with food, they may lack medical supplies.

Keep your response concise—just one or two sentences.
```

### Message 2 — `user`

```text
Previous messages from other people:
Agent 2: I was told: A mudslide just occurred, covering the driveway to North Hill.
Agent 3: I was told: The supply truck headed to the village from East Town was stuck in the tunnel.

It's your turn to speak.
```

## Response contract

```json
{
  "type": "free_text"
}
```

## Block provenance

| Order | Block | Version | Role | Estimated tokens |
|---:|---|---:|---|---:|
| 1 | `scenario_description` | 1 | `system` | 304 |
| 2 | `available_information` | 1 | `system` | 144 |
| 3 | `brevity_instruction` | 1 | `system` | 11 |
| 4 | `discussion_turn` | 1 | `user` | 56 |


---

# HiddenBench paper — one pre-discussion vote

- Prompt version: `hidden_profile_vote_paper@1`
- Messages sent: `2`
- Token counter: `mas_cc_regex_v1_estimate`
- Estimated block tokens: `511`

## Request metadata

```json
{
  "agent_id": 0,
  "audit_answer_included": false,
  "fixture": "hiddenbench_downloaded_task_v1",
  "shuffle_seed": 1026,
  "source_data": "/home/cesarali/LanguageGames/MA-CC/scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/scaled/exact_replication/N_32.json",
  "task_id": 1,
  "transcript_is_inspection_fixture": true
}
```

## Exact messages sent to the LLM

The messages below are shown in transmission order. Text inside each fence is the exact message content.

### Message 1 — `system`

```text
You are participating in a study, acting as a community leader of a small village surrounded by mountains and rivers. Most villagers own cars, but there are also elderly people and children who may need additional assistance when walking. Earlier today, heavy rain began to fall, and the local government issued a warning about a potential disaster.
Hours ago, you requested relief supplies, but the supply truck has yet to arrive. Now, the rain has temporarily stopped, giving you and the others community leaders a short window to decide on the safest evacuation route before the rain resumes. You don’t know how much time you have left to make this critical decision.
Your Task:
you will discuss with other participants, who are also acting as community leaders, to decide where to evacuate. You have three options:
- West City: Accessible through a bridge over the river.
- East Town: Accessible through a tunnel on middle ground.
- North Hill: Accessible through a driveway and walking trails.
Usually, it takes the same time to reach all three places by car, but some routes may be inaccessible now.
There is only one correct evacuation location. After the discussion:
- If you choose the correct location, you will earn $1.
- If all other participants also choose the correct location, you will earn an additional $1 (for a total of $2).
This means that coordinating with others is critical to maximize your rewards. The chat will at most take 15 minutes. However, the exact time when the chat will end is unknown.

You have received the following information. Notice that the order of this information is randomly shuffled; the order of facts does not indicate importance or relationship. Please reason carefully:
- A massive fire has blocked the supply truck and all other traffic.
- The school at North Hill can serve as a temporary evacuation center, providing a two-week supply of essentials and sleeping space in the gym.
- The mayor of East Town has offered accommodations for any evacuees. She also ensures that volunteers are available to assist them.
- The river level is still below the bridge to West City.
- The local government announced that hotels in West City are prepared to accommodate evacuees. While these hotels are fully stocked with food, they may lack medical supplies.

Keep your response concise—just one or two sentences.
```

### Message 2 — `user`

```text
Please decide and provide your rationale in the following JSON format:
{
  "vote": <A string, ["West City", "East Town", "North Hill"]>,
  "rationale": <A string, representing your rationale>
}
```

## Response contract

```json
{
  "allowed_values": [
    "West City",
    "East Town",
    "North Hill"
  ],
  "type": "json_vote"
}
```

## Block provenance

| Order | Block | Version | Role | Estimated tokens |
|---:|---|---:|---|---:|
| 1 | `scenario_description` | 1 | `system` | 304 |
| 2 | `available_information` | 1 | `system` | 144 |
| 3 | `brevity_instruction` | 1 | `system` | 11 |
| 4 | `vote_request` | 1 | `user` | 52 |


---

# HiddenBench paper — one post-discussion vote

- Prompt version: `hidden_profile_vote_paper@1`
- Messages sent: `2`
- Token counter: `mas_cc_regex_v1_estimate`
- Estimated block tokens: `559`

## Request metadata

```json
{
  "agent_id": 0,
  "audit_answer_included": false,
  "fixture": "hiddenbench_downloaded_task_v1",
  "shuffle_seed": 1026,
  "source_data": "/home/cesarali/LanguageGames/MA-CC/scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/scaled/exact_replication/N_32.json",
  "task_id": 1,
  "transcript_is_inspection_fixture": true
}
```

## Exact messages sent to the LLM

The messages below are shown in transmission order. Text inside each fence is the exact message content.

### Message 1 — `system`

```text
You are participating in a study, acting as a community leader of a small village surrounded by mountains and rivers. Most villagers own cars, but there are also elderly people and children who may need additional assistance when walking. Earlier today, heavy rain began to fall, and the local government issued a warning about a potential disaster.
Hours ago, you requested relief supplies, but the supply truck has yet to arrive. Now, the rain has temporarily stopped, giving you and the others community leaders a short window to decide on the safest evacuation route before the rain resumes. You don’t know how much time you have left to make this critical decision.
Your Task:
you will discuss with other participants, who are also acting as community leaders, to decide where to evacuate. You have three options:
- West City: Accessible through a bridge over the river.
- East Town: Accessible through a tunnel on middle ground.
- North Hill: Accessible through a driveway and walking trails.
Usually, it takes the same time to reach all three places by car, but some routes may be inaccessible now.
There is only one correct evacuation location. After the discussion:
- If you choose the correct location, you will earn $1.
- If all other participants also choose the correct location, you will earn an additional $1 (for a total of $2).
This means that coordinating with others is critical to maximize your rewards. The chat will at most take 15 minutes. However, the exact time when the chat will end is unknown.

You have received the following information. Notice that the order of this information is randomly shuffled; the order of facts does not indicate importance or relationship. Please reason carefully:
- A massive fire has blocked the supply truck and all other traffic.
- The school at North Hill can serve as a temporary evacuation center, providing a two-week supply of essentials and sleeping space in the gym.
- The mayor of East Town has offered accommodations for any evacuees. She also ensures that volunteers are available to assist them.
- The river level is still below the bridge to West City.
- The local government announced that hotels in West City are prepared to accommodate evacuees. While these hotels are fully stocked with food, they may lack medical supplies.

Keep your response concise—just one or two sentences.
```

### Message 2 — `user`

```text
Previous messages from other people:
Agent 2: I was told: A mudslide just occurred, covering the driveway to North Hill.
Agent 3: I was told: The supply truck headed to the village from East Town was stuck in the tunnel.

Please decide and provide your rationale in the following JSON format:
{
  "vote": <A string, ["West City", "East Town", "North Hill"]>,
  "rationale": <A string, representing your rationale>
}
```

## Response contract

```json
{
  "allowed_values": [
    "West City",
    "East Town",
    "North Hill"
  ],
  "type": "json_vote"
}
```

## Block provenance

| Order | Block | Version | Role | Estimated tokens |
|---:|---|---:|---|---:|
| 1 | `scenario_description` | 1 | `system` | 304 |
| 2 | `available_information` | 1 | `system` | 144 |
| 3 | `brevity_instruction` | 1 | `system` | 11 |
| 4 | `vote_request` | 1 | `user` | 100 |
