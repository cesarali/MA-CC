# Prompt modifications for the HiddenBench imitation study

Based on `round_019.md` (agent-000, `hidden_bench_imitation_message@1`, gpt-5.4-nano).

Two problems showed up in that log:

1. **The agent never contributed new information.** Every fact in its message was either a shared fact or something it had learned from a partner. Its own private fact (the fire blocking the supply truck) never appeared in any of its five interactions.
2. **The controller announced itself as a controller.** Twice — in the message text and in the history label.

Everything below fixes one of those two things. Parts 1 and 2 are organised by the blocks in your provenance table so they map onto the code. Part 3 is the exception — it covers the simulation loop rather than any prompt text.

A note before starting: these changes affect the **reasoning ON** cells only, since the classical arm has no prompts. That means every prompt edit shifts A and B relative to C and D. Freeze a version before the pilot and report it with the results.

---

## Part 1 — Making information actually surface

### 1.1 `response_style` — remove the length cap

**Current (v1):**

```
Keep your response concise-just one or two sentences.
```

This is probably the single biggest suppressor of information spreading in the whole setup. In two sentences an agent states a conclusion and names one or two supporting facts — and it will pick the most persuasive-sounding ones, which are the shared facts everyone already believes. There is no room to introduce something new *and* argue for it.

**Proposed (v2):**

```
Keep your response short — two to four sentences. Each time you speak,
mention at least one specific piece of information from your list that you
have not already mentioned in an earlier message, and end by saying which
location you are voting for.
```

Two things to know about this:

- The "end by saying which location" part matters for you specifically. This is an imitation model, so the thing being copied is a stated opinion. Right now agents state opinions inconsistently.
- The "not already mentioned" part only works if the agent can see its own past messages. See 1.3.

**This is a real intervention, not a neutral fix.** It is close to the paper's "Share All Information" condition, which moved GPT-4.1 from 0.233 to 0.467. Treat it as a logged prompt version, not a silent default, and keep v1 available so you can report both.

### 1.2 `information` — optionally flag that information is uneven

**Proposed (v2), appended to the fact list:**

```
Other participants have received their own information. Some of what you
know may be known only to you, and some of what they know may be unknown
to you.
```

This is the paper's "Informing Asymmetry" condition (0.367 vs 0.233 baseline). Same treatment as above — a logged version, not a default.

**Do not mark which of the facts are unique.** The hidden-profile paradigm depends on agents not knowing which of their facts are private. Telling them breaks the task. The line above is the strongest hint you can give without destroying the construction.

### 1.3 `private_history` — show the agent its own messages

**Current (v1):**

```
- Event 15: partner/controller said <text>; you committed East Town.
```

The agent sees what others said and what it voted, but never what *it* said. So it has no way to know whether it has already shared something. It cannot follow an instruction like "mention something you haven't mentioned yet," and it cannot notice that it has been silent about its own fact for nineteen rounds.

**Proposed (v2):**

```
- Event 15: the other participant said <text>; you replied <your text>; you committed East Town.
```

Note the label change from `partner/controller` to `the other participant` — that is also fix 2.1 below.

### 1.4 `interaction` — stop nudging toward recycling

**Current (v1):**

```
This is a private exchange with one participant. You may relay information
learned in earlier interactions. Speak now.
```

"Relay information learned" points the agent at things it *heard*. It says nothing about things only it knows. You are actively nudging toward repeating what is already circulating.

**Proposed (v2):**

```
This is a private exchange with one participant. Tell them what you know,
including anything you have not yet told anyone, and say how you are
voting. Speak now.
```

### 1.5 `scenario` — know what the payoff clause is doing

**Current, unchanged:**

```
If all other participants also choose the correct location, you will earn
an additional $1... This means that coordinating with others is critical
to maximize your rewards.
```

This is canonical to the task, so keep it in the baseline. But be aware it is an explicit instruction to conform, sitting in the system prompt, working against everything in Part 1.

For your project this is worth seeing as an opportunity rather than a nuisance: **this paragraph is your imitation coupling strength, written in English.** Classical imitation models have a conformity parameter; here it is a text block. Versioning it gives you a semantic analogue of that parameter to vary later. Worth exposing as `scenario@1` / `scenario@2` (coordination bonus removed) even if you never touch it during the pilot.

---

## Part 2 — Making the controller invisible

### 2.1 Close both leaks

The word "controller" reaches the agent in two places. Both must go.

**Leak 1 — the message text.**

```
The external controller currently advocates option North Hill. Reconsider
your current position before committing your next vote.
```

**Leak 2 — the history renderer.** Every line reads `partner/controller said`. Render every source identically as `the other participant said`, controller and peer alike.

### 2.2 The constraint that shapes everything else

The controller does not receive the hidden facts, and it should stay that way. If it invents evidence, then any movement toward Z just means "an agent believed a new fact" — that is not social control, and it contaminates the truth measurement too.

So the controller has exactly one toolkit: **the shared facts everybody already has, plus a stated vote.** That is enough to be persuasive without polluting anything.

### 2.3 Message structure

Three parts, in order:

1. A reason drawn only from the shared facts
2. A soft coordination appeal
3. An explicit vote

**Z = North Hill:**

> The school at **North Hill** is set up as a proper evacuation center with two weeks of supplies, and the driveway is still open even with the mudslide — most people have cars anyway. That seems like the safest option for everyone. I'm voting **North Hill**.

**Z = East Town:**

> **East Town** has the mayor's accommodations and volunteers ready for the elderly and children, and the tunnel is on middle ground. I think that's the most reliable choice. I'm voting **East Town**.

**Z = West City:**

> The hotels in **West City** are already stocked with food and medical supplies, which matters most for the elderly and the kids. I'm voting **West City**.

Every fact in those comes from the shared block. Nothing invented.

### 2.4 Style rules to enforce in code

- Two to four sentences — the same range the agents have
- Bold the option name, since the peers do
- First person, ending with "I'm voting **Z**"
- Never the words *controller, system, experiment, external, your position, reconsider, commit*
- Never an instruction — a peer states its view, it does not direct anyone

### 2.5 Use a small paraphrase bank

Agent-000 sees five past events at once. The identical sentence three times reads as a bot even when the wording is natural. Use three or four variants per (task, Z), each with a stable ID, and log which one fired. Then you can check that no single variant is carrying the whole effect.

---

## Part 3 — Event scheduling (code, not prompts)

This is the one section that is not about prompt text. Whether the controller **replaces** a peer conversation or **adds** an extra one is decided by the simulation loop, not by anything written in a message. No prompt string can enforce it.

### 3.1 Replace, don't add

Two ways a control event can work:

- **Replace** a peer conversation → every agent always has the same number of conversations. Only the content of one differs.
- **Add** an extra conversation → the controlled group now has more conversations than the uncontrolled group, and any difference might just be "more talking happened."

Replacement is the right choice. The cost is that every time the controller speaks, a real peer conversation did not happen — and peer conversations are where facts spread. So control quietly slows information spreading down.

That cost is measurable, though. With evidence reach logged (Part 4), you can check afterwards whether control hurt truth by being pushy or just by eating conversations. Unequal conversation counts, by contrast, are baked into the design and cannot be untangled later. Prefer the problem you can measure.

### 3.2 This is already your stated design — for one arm

The weekly report, Section 3:

> "In reasoning mode, ADVOCATE Z **replaces** an ordinary local interaction by a fixed controller message; the focal LLM may accept or reject it."

Good. Nothing to change there.

### 3.3 Open question: what happens in classical mode

The next sentence in the same paragraph is ambiguous:

> "In classical mode, the same control event **modifies the transition weight** toward Z in a provider-free way."

"Modifies the transition weight" does not clearly mean replacement. It reads more like tilting the odds on an interaction that still happens.

If that is what the code does, then a control event consumes a peer interaction in the reasoning arm but not in the classical arm — and **B − D is confounded**, because the two arms have different event counts under control. It probably does not distort the classical dynamics much, since there is no evidence to lose there, only opinions. But the two arms would no longer be matched on the thing the design claims to match them on.

Resolve the wording, then make the code match it. Whichever way you go, both arms must go the same way.

### 3.4 Invariants to assert per episode

- For the same seed, **total events per agent is identical across all four cells** (A, B, C, D)
- In a controlled episode, `peer_interactions == total_events − control_events`, and this identity holds in **both** reasoning and classical mode
- The **event sequence replays**: same focal agent, same partner, same order, across ON and OFF

The third one goes beyond what the report currently promises. Section 6.1 lists "the same explicit X₀ can be replayed in reasoning and classical modes" as a platform success criterion — but replaying X₀ is not sufficient. If the interaction schedule differs between arms, matched initial conditions do not buy you a matched comparison.

---

## Part 4 — What to log

Most of this exists to let you tell apart explanations after the fact.

| Log | Why |
|---|---|
| Evidence reach: which agent has heard which fact, per event | The central measurement. Tells you whether facts spread at all, and lets you separate "control starved the conversation" from "control was adversarial" |
| Disclosure events: which fact entered circulation, from whom, when | Gives you shared-vs-unshared diffusion curves — the cheapest real result available to you |
| Controller template ID per control event | Rules out one paraphrase driving everything |
| Vote entropy / `H(n_t)` at every event | Distinguishes "sensor broken" from "population frozen, nothing to sense" |
| The agent's own message text, per event | Needed for 1.3, and for detecting which facts were cited |

On detecting cited facts: keep the agent's message natural and run fact-detection as a **separate pass afterwards**. Do not ask the agent to list the facts it used in its own response — that would itself increase disclosure and contaminate the thing you are measuring.

---

## Part 5 — Three checks before the real pilot

**5.1 Reveal-All on your own model.** Give every agent all eleven facts at the start, no discussion, and see whether nano picks North Hill. The paper got 0.967 with GPT-4.1, but measured GPT-5-Nano as the weakest model in the study — communication improved it by −0.004. If your model scores near the floor here, truth-convergence is impossible for reasons that have nothing to do with your controller or your dynamics, and you should switch models before spending anything else. **This is the run that decides whether the rest of the week is interpretable.**

**5.2 Manipulation check on the controller.** Take twenty controller messages and twenty peer messages, strip the labels, and ask a fresh model to guess which is which. Above chance means the controller is still detectable, and `R_ctrl` is measuring "agent discounts an obvious outside voice" rather than social control.

**5.3 Check the population is not frozen.** Agent-000 committed East Town at events 1, 3, 6, 10 and 15 — it never moved once. If `H(n_t)` is near zero across the board, every mutual information estimate will be near zero because the state has no variation, not because the sensor or the controller failed. Confirm there is movement to measure before interpreting any information quantity.

---

## Part 6 — The third controller arm

Once the controller passes as a peer, add a third column to the design: same structure, same length, same style, but advocating a **random non-target option**.

The paper found that a single dissenting agent nearly doubles accuracy (0.233 → 0.492) more or less regardless of what it argues for — simply by breaking premature consensus. Your controller is structurally that agent. Without this arm you cannot tell directed control from generic dissent, and the wrong-target reversal alone will not separate them, because reversal injects dissent too.
