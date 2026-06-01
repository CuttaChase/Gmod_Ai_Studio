# Garry's Mod Lua Training Prompts

Use these prompts to generate new high-quality question/answer rows for your training dataset.

Recommended answer format:

- Explain the reasoning briefly.
- State the correct realm behavior: server, client, or shared.
- Give working Garry's Mod Lua code.
- Mention validation, prediction, networking, or persistence concerns when relevant.

How to use them for training:

1. Pick a prompt.
2. Write or generate a strong answer and fix it by hand.
3. Save it as a `question` and `answer` row in your JSONL dataset.
4. Retrain on the updated dataset.

## Addon Structure And Realms

1. Build a minimal Garry's Mod addon skeleton with `autorun`, shared, server, and client files, and explain what each file is responsible for.
2. Refactor a messy one-file Garry's Mod addon into a clean file layout using `shared.lua`, `init.lua`, and `cl_init.lua`.
3. Explain the difference between `include`, `AddCSLuaFile`, and shared file loading in Garry's Mod Lua, with an example addon bootstrap.
4. Show how to structure a large addon so configuration, networking, persistence, entities, and UI stay separated.
5. Explain what kinds of logic must stay on the server, what belongs on the client, and what is safe to keep shared.
6. Build a shared utility module for a Garry's Mod addon and show how server and client code should use it safely.

## Debugging And Investigation

7. Show a practical debugging workflow for a Garry's Mod addon that only breaks on a dedicated server.
8. Write a clean debug logger helper for Garry's Mod Lua that prefixes messages and makes it obvious which subsystem produced them.
9. Debug a Garry's Mod hook that appears to run twice and explain how prediction or duplicate hook registration can cause that.
10. Show how to inspect installed hooks and trace which function is overriding expected behavior in Garry's Mod Lua.
11. Explain how to debug invalid entity errors safely using `IsValid`, print statements, and world-state checks.
12. Show how to debug a networking bug where the client HUD never receives the server value it expects.
13. Explain how to debug timers that are duplicated, never removed, or silently keep running after an entity is gone.
14. Show how to debug why a scripted entity works in sandbox but fails after a map cleanup or server restart.

## Networking And State Sync

15. Build a Garry's Mod net message flow where the server sends inventory data to the client and the client renders it in a menu.
16. Show a secure request-response pattern using `net.Start`, `net.Receive`, and server-side validation for a purchase system.
17. Explain how to design a compact network protocol for syncing player stats without spamming giant tables.
18. Build a Garry's Mod example where a server-owned currency value is synchronized to a client HUD and updated live.
19. Show how to use `SetupDataTables` to network entity state instead of manually sending every update.
20. Explain when to use networked vars, when to use `net` messages, and when not to sync data at all.
21. Show how to rate-limit a net message handler to stop players from spamming expensive server logic.
22. Build a client request that asks the server to perform an action, with server-side range, ownership, and permission checks.

## Players, Entities, And World Interaction

23. Explain the practical difference between `Player`, `Entity`, `Weapon`, `NPC`, and `Vehicle` methods in Garry's Mod Lua.
24. Build a scripted entity that stores its owner, uses `SetupDataTables`, and lets players interact with it safely.
25. Show how to create a spawnable entity with physics setup, `Use` interaction, and cleanup-safe behavior.
26. Build a trace-based interaction system where a player can look at an entity and trigger a contextual action.
27. Explain how to use `util.TraceLine` and `util.TraceHull` for toolgun-like interactions and custom abilities.
28. Show how to create an entity that updates a 3D2D label client-side but keeps the real state server-side.
29. Build a simple ownership system for spawned entities keyed to `SteamID64` and explain how to validate it.
30. Show how to make a Garry's Mod entity save custom state and restore it after a restart or map reload.

## SWEPs And Weapons

31. Build a SWEP with proper shared, server, and client file splitting and explain what belongs in each file.
32. Show a prediction-safe `PrimaryAttack` implementation that gives responsive effects but keeps damage authoritative on the server.
33. Build a SWEP reload system with cooldowns, animation, ammo checks, and anti-spam logic.
34. Explain how to design a SWEP that has client-side HUD feedback and server-side damage logic without desync.
35. Build a SWEP secondary fire that launches a scripted entity projectile and validates ammo usage correctly.
36. Show how to make a custom melee SWEP that uses traces, lag compensation awareness, and server validation.
37. Explain how to add weapon upgrades or attachments in an addon-friendly way without hardcoding everything into one file.
38. Build a weapon ability system with cooldown UI, prediction-safe effects, and persistence for unlocked upgrades.

## Hooks And Gamemode Flow

39. Explain how to choose the right hook in Garry's Mod when you want to modify spawning, damage, movement, or UI.
40. Build a system using `PlayerInitialSpawn`, `PlayerSpawn`, and `PlayerDisconnected` to load, initialize, and save player data.
41. Show how to use `GM:PlayerSpawn` and related hooks to implement a custom loadout system.
42. Explain how to safely override gamemode behavior without breaking unrelated hooks from other addons.
43. Build a round-based system that uses hooks to start rounds, end rounds, respawn players, and clean temporary entities.
44. Show how to use movement hooks like `SetupMove` or `Move` to create a custom dash or stamina mechanic.
45. Explain how to write hooks so they are easy to remove, debug, and isolate by subsystem.
46. Build a custom event system inside an addon that wraps several Garry's Mod hooks into a clean subsystem API.

## Persistence, Save Data, And Databases

47. Show how to save per-player addon data as JSON files in `garrysmod/data` using `SteamID64` as the key.
48. Explain when to use `Player:SetPData`, JSON files, `sql.Query`, or client cookies for Garry's Mod persistence.
49. Build a persistence layer for a player economy system and explain how to avoid data loss on disconnect or shutdown.
50. Show how to save a structured inventory table using `util.TableToJSON` and `util.JSONToTable` safely.
51. Build a SQLite-backed Garry's Mod addon subsystem and explain how to sanitize strings with `sql.SQLStr`.
52. Explain how to version or migrate saved addon data when the schema changes between releases.
53. Show how to persist world-owned entities or bases and restore them after a server restart.
54. Build a save system that batches writes or schedules them to reduce hitching from constant disk access.

## VGUI, HUD, And Menus

55. Build a Garry's Mod admin-style menu with categories, reusable panels, and client-side state refresh from net updates.
56. Explain how to structure large VGUI code so layout, rendering, and data sync do not get mixed together.
57. Build an inventory UI that renders client-side copies of server-owned item data and updates cleanly when new data arrives.
58. Show how to create a custom `DFrame` workflow with tabs, item rows, confirm dialogs, and a reusable theme.
59. Build a HUD element that shows player stamina, cooldowns, and context-sensitive information from synchronized server state.
60. Explain how to make VGUI panels reusable across multiple addon systems instead of rebuilding everything inline.
61. Show how to use `DModelPanel` or 3D previews inside an equipment or loadout menu.
62. Build a spawnmenu extension or creation tab for a custom addon system.

## Rendering, 3D2D, And Visual Effects

63. Explain how to choose between `HUDPaint`, `ENT:Draw`, and world render hooks for Garry's Mod visuals.
64. Build a client-side 3D2D world label that displays entity state above a scripted entity.
65. Show how to render a beam, sprite, or simple world marker from synchronized server state.
66. Explain how to structure client rendering code so expensive drawing is gated by distance or relevance.
67. Build a Garry's Mod effect or marker system that highlights valid interactable entities in the world.
68. Show how to draw custom HUD indicators for objective locations or teammates.
69. Explain how to combine 2D HUD feedback with 3D world indicators without duplicating state logic.
70. Build a client visual debugging helper using `debugoverlay` or lightweight render markers.

## Prediction, Movement, And Complex Interaction

71. Explain prediction in Garry's Mod Lua and show how to avoid duplicated sounds, particles, or weapon logic.
72. Build a prediction-safe sprint or dash mechanic with server validation and client HUD feedback.
73. Show how to use movement hooks to implement a wall jump, slide, or parkour-style system.
74. Explain which parts of movement and combat code should be predicted and which should remain server-only.
75. Build a usable entity or ability that feels responsive locally but still validates range and permissions on the server.
76. Show how to fix a predicted system that applies ammo use, cooldowns, or effects twice.

## Performance And Security

77. Explain the biggest performance mistakes in Garry's Mod addons and how to avoid them.
78. Build a subsystem that avoids expensive `Think` loops by using timers, hooks, and cached values.
79. Show how to profile or reason about a slow addon that scans all entities every frame.
80. Explain how to secure a Garry's Mod net message so players cannot fake prices, damage, or entity ownership.
81. Build a server-side validation layer for a crafting or purchase system that rejects malformed client requests.
82. Show how to keep rendering, networking, and persistence from becoming bottlenecks in a large addon.
83. Explain how to design systems so they degrade gracefully with many players or many world entities.
84. Build a cleanup-safe subsystem that removes hooks, timers, and references when an entity is deleted.

## Complex Systems And Creative Ideas

85. Design a Garry's Mod base-building addon with power nodes, generators, storage, defense turrets, and persistence.
86. Design a survival system with hunger, temperature, crafting, resource gathering, world entities, and save data.
87. Build the architecture for a detective or evidence system with traces, world clues, UI, and round cleanup.
88. Design a faction or territory control system that combines player data, capture zones, HUD, and persistence.
89. Build a quest system with objectives, NPC interaction, progress saving, and client HUD updates.
90. Design a vehicle progression addon with garages, upgrades, persistence, and custom VGUI.
91. Build an advanced inventory and equipment system with stacking items, loadouts, rarity, and synchronized UI.
92. Design a cooperative PvE addon with NPC waves, dynamic objectives, world pickups, and team rewards.
93. Build the core architecture for a job or role system with permissions, spawn logic, salaries, and persistence.
94. Design a roguelite-style Garry's Mod mode with run-based upgrades, random events, save checkpoints, and reward systems.
95. Build a magic or ability framework with cooldowns, prediction-safe casting, world effects, and server-owned combat rules.
96. Design a creative addon idea that combines hooks, entities, SWEPs, VGUI, persistence, and networking into one cohesive system.

## Refactor And Repair Prompts

97. Refactor a Garry's Mod addon that mixes client and server code in one file into a correct multi-file layout.
98. Repair a scripted entity that throws invalid entity errors after cleanup and explain the root cause.
99. Refactor a spammy `Think`-based addon into an event-driven architecture with timers and targeted hooks.
100. Fix a Garry's Mod addon that stores the real inventory only on the client and redesign it as a secure server-owned system.
101. Repair a SWEP that double-fires sounds and particles because of prediction mistakes.
102. Refactor a huge VGUI function into reusable panel classes and explain why the result is easier to maintain.
103. Repair a save system that loses player progress on crash or restart and propose safer write timing.
104. Fix a net message handler that trusts client values directly and rewrite it with proper server validation.