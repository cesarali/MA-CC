# HiddenBench raw-data task summary

## Scope

This report describes the unchanged raw source file:

`scripts/local_llms/hiddenbench_population_pipeline/data/hiddenbench/source/benchmark.json`

It does **not** use the canonicalized tasks, expanded populations, generated
paraphrases, or experiment results. The accompanying metadata identifies the
source as `YuxuanLi1225/HiddenBench` at commit
`1e3c25b1fd798c6717f4df0463edd3825c8e37f9`. The raw file has SHA-256
`2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3`.

## What one task contains

Every task has:

- a numeric `id` and a task `name`;
- a scenario description;
- shared clues, which all participants can see;
- hidden clues, which are meant to be distributed across participants;
- a list of possible answers; and
- one labeled correct answer, which always exactly matches one of the options.

Tasks 9–65 also have a `rationale` explaining the intended puzzle design. Tasks
1–8 do not. The report counts what is actually stored, rather than treating the
rationale as proof that every puzzle is logically valid.

## Dataset at a glance

| Characteristic | Raw-data result |
| --- | --- |
| Number of tasks | 65 |
| Task IDs | All integers from 1 through 65, with no gaps |
| Unique task names | 65 |
| Tasks with 3 answer options | 59 |
| Tasks with 4 answer options | 6 (IDs 4, 6, 17, 25, 38, and 53) |
| Shared-clue items | 250 total; 3–8 per task; median 4 |
| Hidden-clue items | 253 total; 3–4 per task |
| Tasks with 4 hidden clues | 58 |
| Tasks with 3 hidden clues | 7 (IDs 4, 6, 8, 30, 33, 34, and 50) |
| Tasks with a design rationale | 57 (IDs 9–65) |
| Correct answer position | option 1: 11 tasks; option 2: 13; option 3: 39; option 4: 2 |

The typical task is therefore a three-option group decision with four shared
clues and four separately held hidden clues. Most scenarios use an attractive
but wrong initial choice: participants must pool their private facts to rule out
the decoy and identify the only workable answer. Common settings include
evacuation, shelter selection, logistics, research sites, secure storage, and
mystery/deduction problems.

The answer labels are not balanced across positions: 39 of 65 correct answers
(60%) are the third option. This is a relevant property for evaluations because
a model could benefit from answer-position bias.

## How the text counts were calculated

The item counts are exact JSON list lengths. Sentence counts are reproducible
estimates: whitespace was normalized, common abbreviations such as `Mr.` and
`Dr.` were protected, and text was split after `.`, `?`, or `!`. Headings and
list fragments in descriptions do not always end in punctuation, so description
sentence counts should be read as approximate rather than linguistic ground
truth.

Across the complete file:

| Text field | Total sentences | Mean sentences per task | Sentence range per task | Total words | Mean words per task |
| --- | ---: | ---: | ---: | ---: | ---: |
| Description | 481 | 7.4 | 3–17 | 7,960 | 122.5 |
| Shared clues | 294 | 4.5 | 3–11 | 5,190 | 79.8 |
| Hidden clues | 279 | 4.3 | 3–9 | 6,551 | 100.8 |
| Rationale (57 tasks only) | 451 | 7.9 | — | 10,424 | 182.9 |

In the per-task tables, `4 / 6` under “Shared items / sentences,” for example,
means four separately stored shared clues containing about six sentences in
total. “Answer pos.” is the one-based position of the correct answer in the
option list.

## Per-task summary: tasks 1–33

| ID | Task and decision options | Correct answer (position) | Options | Shared items / sentences | Hidden items / sentences | Description sentences |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 1 | `evacuation_west_city`: West City; East Town; North Hill | West City (1) | 3 | 4 / 6 | 4 / 4 | 17 |
| 2 | `evacuation_north_hill`: West City; East Town; North Hill | North Hill (3) | 3 | 4 / 7 | 4 / 4 | 17 |
| 3 | `evacuation_east_town`: West City; East Town; North Hill | East Town (2) | 3 | 4 / 6 | 4 / 4 | 17 |
| 4 | `toma_butera_2009`: Mr. X; Mr. X's son; Mrs. Y; Mr. Z | Mr. X's son (2) | 4 | 3 / 6 | 3 / 9 | 11 |
| 5 | `baker_2010`: Stevens; Roberts; Jones | Roberts (2) | 3 | 6 / 7 | 4 / 4 | 15 |
| 6 | `schulz_hardt_mojzisch_2012`: Candidates A–D | Candidate C (3) | 4 | 4 / 4 | 3 / 3 | 8 |
| 7 | `graetz_et_al_1998`: Franklin Enterprises; Starlight Incorporated; Cape Industries | Starlight Incorporated (2) | 3 | 3 / 3 | 4 / 4 | 17 |
| 8 | `Stasser_Stewart_1992`: Eddie Sullivan; Billy Prentice; Mickey Malone | Eddie Sullivan (1) | 3 | 8 / 8 | 3 / 9 | 13 |
| 9 | `critical_hospital_transfer`: Hospitals A–C | Hospital A (1) | 3 | 3 / 6 | 4 / 4 | 8 |
| 10 | `emergency_supply_drop`: Warehouses A–C | Warehouse C (3) | 3 | 3 / 3 | 4 / 4 | 6 |
| 11 | `emergency_conference_relocation`: City Library; Community Center; School Gym | School Gym (3) | 3 | 4 / 6 | 4 / 5 | 6 |
| 12 | `evacuate_park_dilemma`: Blueberry Ridge; Green Valley; Red Lake | Green Valley (2) | 3 | 4 / 5 | 4 / 5 | 7 |
| 13 | `Laboratory Theft Deduction`: Labs Alpha–Gamma | Lab Gamma (3) | 3 | 3 / 4 | 4 / 4 | 8 |
| 14 | `lunch_group_decision`: Restaurants A–C | Restaurant C (3) | 3 | 3 / 3 | 4 / 4 | 5 |
| 15 | `artifact_safe_haven`: City Bank; University Library; Downtown Police Station | Downtown Police Station (3) | 3 | 3 / 3 | 4 / 4 | 7 |
| 16 | `Crisis Backup Decision`: Alpha; Bravo; Charlie | Charlie (3) | 3 | 4 / 4 | 4 / 4 | 6 |
| 17 | `scientists_animal_base_decision`: Deep Jungle; Riverbank; Hilltop; Plateau | Plateau Camp (D) (4) | 4 | 4 / 4 | 4 / 4 | 8 |
| 18 | `choosing_base_camp`: Camp Summit; Camp Pinecone; Camp Meadow | Camp Pinecone (2) | 3 | 4 / 4 | 4 / 4 | 5 |
| 19 | `city_storm_shelter_decision`: Greenfield High; Blue River Center; Oakridge Library | Oakridge Library (3) | 3 | 3 / 5 | 4 / 5 | 6 |
| 20 | `meteor_shower_shelter`: Shelters Alpha–Gamma | Shelter Alpha (1) | 3 | 3 / 3 | 4 / 4 | 7 |
| 21 | `emergency_transportation_decision`: train station; bus terminal; airstrip | Celestia Airstrip (3) | 3 | 3 / 3 | 4 / 6 | 6 |
| 22 | `Antarctic Storm Safe Haven`: Main Base; Retreat Camp; Rescue Outpost | Main Base (1) | 3 | 4 / 4 | 4 / 4 | 10 |
| 23 | `community_banquet_venue_decision`: Lakeview Resort; Grand Oak Hotel; Heritage Library | Grand Oak Hotel (2) | 3 | 4 / 4 | 4 / 4 | 7 |
| 24 | `Critical Data Backup Site Selection`: server farm; research annex; city HQ | City HQ Facility (3) | 3 | 3 / 3 | 4 / 4 | 8 |
| 25 | `select_emergency_shelter`: Stations Alpha–Delta | Station Delta (4) | 4 | 4 / 6 | 4 / 6 | 5 |
| 26 | `manuscript_flood_shelter`: science library; chapel; town hall | Town Hall (3) | 3 | 3 / 3 | 4 / 5 | 6 |
| 27 | `research_station_site_selection`: Maple Valley; Copper Lake; Pine Ridge | Pine Ridge (3) | 3 | 3 / 5 | 4 / 4 | 5 |
| 28 | `Rescue the Lost Researchers`: River Route; Mountain Pass; Forest Trail | Forest Trail (3) | 3 | 4 / 4 | 4 / 4 | 6 |
| 29 | `Safe Shelter Selection`: Mountain Lodge; Riverside Park; Summerfield School | Riverside Park (2) | 3 | 4 / 4 | 4 / 4 | 6 |
| 30 | `the_lead_investor_decision`: Peak Capital; Skylake Ventures; Northstar Partners | Skylake Ventures (2) | 3 | 3 / 3 | 3 / 3 | 6 |
| 31 | `weather_sensor_deployment`: Alpha Ridge; Beta Valley; Gamma Lake | Gamma Lake (3) | 3 | 3 / 4 | 4 / 5 | 6 |
| 32 | `critical_vaccine_route`: Mountain Pass; Frozen Lake; Old Supply Road | Route C, Old Supply Road (3) | 3 | 4 / 7 | 4 / 4 | 6 |
| 33 | `Critical Sample Transfer`: Lab Gemini; Lab Atlas; Lab Nova | Lab Nova (3) | 3 | 3 / 3 | 3 / 3 | 4 |

## Per-task summary: tasks 34–65

| ID | Task and decision options | Correct answer (position) | Options | Shared items / sentences | Hidden items / sentences | Description sentences |
| ---: | --- | --- | ---: | ---: | ---: | ---: |
| 34 | `Safe Haven After the Spill`: Maple Lodge; Pine Retreat; Cedar Station | Cedar Station (3) | 3 | 4 / 4 | 3 / 3 | 6 |
| 35 | `the_safe_shelter`: Red House; Blue House; Green House | Red House (1) | 3 | 4 / 4 | 4 / 4 | 7 |
| 36 | `datacenter_emergency_migration`: Datacenters Alpha–Gamma | Datacenter Gamma (3) | 3 | 4 / 4 | 4 / 4 | 7 |
| 37 | `emergency_warehouse_selection`: Bayview; Central; Hilltop | Hilltop Storage (3) | 3 | 4 / 4 | 4 / 4 | 8 |
| 38 | `storm_recovery_clinic_site_selection`: open field; sports center; hilltop park; library | Site C, Hilltop park (3) | 4 | 4 / 11 | 4 / 4 | 7 |
| 39 | `emergency_aircraft_landing_site`: LZ Alpha; LZ Bravo; LZ Charlie | LZ Alpha (1) | 3 | 5 / 6 | 4 / 4 | 8 |
| 40 | `emergency_hospital_transfer`: Lakeview; Riverbend; Meadow Valley | Meadow Valley General (3) | 3 | 4 / 4 | 4 / 4 | 6 |
| 41 | `Space Evacuation Decision`: Orion Station; Lunar Base; Mars Colony | Mars Colony (3) | 3 | 4 / 4 | 4 / 4 | 5 |
| 42 | `safe_haven_decision`: Mason's Cabin; Riverside Shelter; Eagle Ridge Lodge | Eagle Ridge Lodge (3) | 3 | 4 / 5 | 4 / 4 | 8 |
| 43 | `The Artifact Delivery`: Aurora Station; Beacon Post; Canyon Depot | Canyon Depot (3) | 3 | 4 / 4 | 4 / 4 | 5 |
| 44 | `Choosing the Safe Offsite Venue`: Downtown Hotel; Riverside Center; Hilltop Retreat | Hilltop Retreat (3) | 3 | 4 / 4 | 4 / 5 | 6 |
| 45 | `Safe Lab Choice After Earthquake`: Labs Alpha–Gamma | Lab Gamma (3) | 3 | 4 / 4 | 4 / 4 | 8 |
| 46 | `emergency_evacuation_center_choice`: school gym; community center; library hall | Riverside School Gym (1) | 3 | 5 / 5 | 4 / 4 | 3 |
| 47 | `Find the Missing Prototype`: conference room; storage room; CEO's office | CEO's Office (3) | 3 | 4 / 4 | 4 / 4 | 7 |
| 48 | `missing_lab_sample`: Cold Storage B; Lab 2 Bench; Decontamination Room | Lab 2 Bench (2) | 3 | 4 / 4 | 4 / 4 | 7 |
| 49 | `choosing_the_safe_field_station`: Riverside; Hilltop; Valley Edge | Hilltop Cabin (2) | 3 | 4 / 4 | 4 / 4 | 5 |
| 50 | `secure_meeting_room_decision`: Conference Rooms Alpha–Gamma | Conference Room Gamma (3) | 3 | 3 / 4 | 3 / 3 | 6 |
| 51 | `emergency_supply_distribution`: Alpha; Bravo; Charlie Storage | Charlie Storage (3) | 3 | 4 / 4 | 4 / 4 | 10 |
| 52 | `mountain_storm_shelter`: Blue Lake; Red Cliff; Green Valley | Green Valley Outpost (3) | 3 | 4 / 5 | 4 / 4 | 6 |
| 53 | `sensor_placement_decision`: Ocean Bluff; Mountain Ridge; River Valley; Forest Edge | Mountain Ridge (2) | 4 | 4 / 4 | 4 / 4 | 6 |
| 54 | `island_research_base_choice`: Sites A–C | Site C (3) | 3 | 4 / 4 | 4 / 4 | 7 |
| 55 | `emergency_drone_delivery`: Park; School; Hospital | Hospital (3) | 3 | 4 / 4 | 4 / 5 | 7 |
| 56 | `Secure the Masterpiece`: government vault; art museum storage; university lab | Government Records Vault (1) | 3 | 5 / 5 | 4 / 4 | 7 |
| 57 | `archaeological_dig_site`: Sites A–C | Site C (3) | 3 | 4 / 4 | 4 / 4 | 4 |
| 58 | `secure_negotiation_site_selection`: conference center; city hall; museum basement | Old Museum Basement (3) | 3 | 3 / 3 | 4 / 4 | 4 |
| 59 | `last_minute_move`: Alpha Movers; Bravo Moving; Charlie's Transport | Bravo Moving Co. (2) | 3 | 4 / 5 | 4 / 5 | 6 |
| 60 | `The Elusive Bird Sighting`: Oak Woods; Wetland; Meadow | Area A, Oak Woods (1) | 3 | 4 / 4 | 4 / 5 | 4 |
| 61 | `power_outage_island`: generator room; water pump; communication tower | Communication Tower (3) | 3 | 4 / 5 | 4 / 4 | 7 |
| 62 | `company_acquisition_decision`: biomedical; AI hardware; logistics software | Logistics software company (3) | 3 | 4 / 4 | 4 / 4 | 9 |
| 63 | `Emergency Event Relocation`: Garden Hall; River Pavilion; Mountain Lodge | Garden Hall (1) | 3 | 3 / 6 | 4 / 5 | 6 |
| 64 | `Office Outbreak Mystery`: food; water; airborne HVAC toxin | Airborne Toxin from HVAC (3) | 3 | 4 / 4 | 4 / 4 | 6 |
| 65 | `Missing Medicine Delivery`: Pharmacy; Clinic; School Gym | School Gym (3) | 3 | 4 / 4 | 4 / 4 | 8 |

## Notable raw-data characteristics

- IDs 1–3 reuse the same 256-word evacuation description and the same three
  destinations, but change the evidence and correct destination. This is the
  only description repeated in the file; it appears three times.
- Six tasks have four options, contrary to the existing provenance statement
  that every task has exactly three.
- Seven tasks have three hidden clue items; the other 58 have four. A hidden
  clue count should therefore be read from each record rather than hard-coded.
- Shared-clue length varies more widely than hidden-clue length. Task 8 has
  eight shared items, while task 5 has six; all remaining tasks have three to
  five.
- Naming style is inconsistent but IDs are stable: some names use
  `snake_case`, while others use spaces, title case, or capital letters.
- The raw file contains the labeled answers and, for most tasks, long design
  rationales. These fields must not be exposed to agents during a blind
  benchmark run.

This is a structural and descriptive inspection of the raw records. It does
not test whether every clue is necessary, whether each private view is truly
insufficient, or whether every rationale agrees perfectly with its task text;
those require a separate semantic-validity audit.
