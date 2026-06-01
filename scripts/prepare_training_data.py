#!/usr/bin/env python3
import csv
import html
import json
import pathlib
import random
import re
import textwrap


REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
DATASETS_DIR = REPO_ROOT / "datasets"
SCRAPED_DATA_DIR = DATASETS_DIR / "scraped"
UPLOADED_FILES_TRAIN_PATH = DATASETS_DIR / "uploaded_files_train.jsonl"
LEGACY_GMOD_SOURCE = DATASETS_DIR / "gmod_coding_dataset.txt"
LEGACY_LUA_SOURCE = DATASETS_DIR / "5.1.txt"
RANDOM_SEED = 42
CORE_ADDON_API_TITLES = {
    "hook.Add",
    "hook.Call",
    "hook.Run",
    "net.Receive",
    "net.Start",
    "net.WriteBool",
    "net.WriteString",
    "net.WriteUInt",
    "net.ReadBool",
    "net.ReadString",
    "net.ReadUInt",
    "util.AddNetworkString",
    "util.TableToJSON",
    "util.JSONToTable",
    "ents.Create",
    "scripted_ents.Register",
    "weapons.Register",
    "concommand.Add",
    "timer.Create",
    "timer.Simple",
    "vgui.Create",
    "include",
    "AddCSLuaFile",
    "file.CreateDir",
    "file.Exists",
    "file.Read",
    "file.Write",
    "Player:GetPData",
    "Player:SetPData",
    "Player:RemovePData",
    "sql.Query",
    "sql.SQLStr",
    "http.Fetch",
    "http.Post",
    "util.TraceLine",
    "util.TraceHull",
    "util.Compress",
    "util.Decompress",
    "debug.Trace",
    "debugoverlay.Line",
    "debugoverlay.Text",
    "PrintTable",
    "ErrorNoHalt",
    "MsgN",
    "IsFirstTimePredicted",
    "cam.Start3D",
    "cam.End3D",
    "cam.Start3D2D",
    "cam.End3D2D",
    "render.SetMaterial",
    "render.DrawSprite",
    "render.DrawBeam",
    "surface.SetDrawColor",
    "surface.DrawRect",
    "undo.Create",
    "cleanup.Add",
    "duplicator.RegisterEntityClass",
    "resource.AddWorkshop",
    "surface.CreateFont",
}
ENTITY_METHOD_PREFIXES = ("Entity:", "Player:", "Weapon:", "NPC:", "NextBot:", "Vehicle:")
ENTITY_HOOK_PREFIXES = ("ENT:", "ENTITY:")
HOOK_TITLE_PREFIXES = ("GM:", "GAMEMODE:", "WEAPON:", "ENTITY:", "ENT:", "TOOL:", "PANEL:", "EFFECT:", "DRIVE:")
SWEP_PREFIXES = ("WEAPON:", "SWEP")
PANEL_PREFIXES = ("Panel:", "PANEL:")
ADDON_CODE_PATTERN = re.compile(
    r"(hook\.Add|net\.(Start|Receive|WriteBool|WriteString|WriteUInt|ReadBool|ReadString|ReadUInt)|util\.(AddNetworkString|TableToJSON|JSONToTable|TraceLine|TraceHull|Compress|Decompress)|ents\.Create|scripted_ents\.Register|weapons\.Register|vgui\.Create|concommand\.Add|timer\.(Create|Simple)|AddCSLuaFile|include\(|surface\.(CreateFont|SetDrawColor|DrawRect)|file\.(CreateDir|Exists|Read|Write)|Player:(GetPData|SetPData|RemovePData)|sql\.(Query|SQLStr)|http\.(Fetch|Post)|debugoverlay\.(Line|Text)|ErrorNoHalt|PrintTable|MsgN|IsFirstTimePredicted|cam\.(Start3D|End3D|Start3D2D|End3D2D)|render\.(SetMaterial|DrawSprite|DrawBeam)|undo\.Create|cleanup\.Add|duplicator\.RegisterEntityClass|resource\.AddWorkshop)",
    re.I,
)
CURATED_GMOD_CONCEPTS = [
    {
        "title": "Addon Structure And Realms",
        "question": "How should I structure a Garry's Mod addon so shared, server, and client code stay organized?",
        "answer": textwrap.dedent(
            """\
            Treat a Garry's Mod addon as a small Lua project with clear realm boundaries.

            Rules
            - Put authority, spawning, damage, persistence, and validation on the server.
            - Put HUD, VGUI, rendering, and client-only effects on the client.
            - Put constants, shared helpers, and network message names in shared files.
            - Send client files from the server with AddCSLuaFile before clients need them.
            - Use include to load modules explicitly instead of putting all logic in one file.

            Common layout
            - lua/autorun/myaddon.lua
            - lua/autorun/server/sv_myaddon.lua
            - lua/autorun/client/cl_myaddon.lua
            - lua/myaddon/sh_config.lua
            - lua/entities/sent_example/shared.lua
            - lua/entities/sent_example/init.lua
            - lua/entities/sent_example/cl_init.lua
            - lua/weapons/weapon_example/shared.lua
            - lua/weapons/weapon_example/init.lua
            - lua/weapons/weapon_example/cl_init.lua

            The shared bootstrap usually decides which files to send and include. Scripted entities and SWEPs usually keep shared definitions in shared.lua, server logic in init.lua, and client drawing code in cl_init.lua.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook",
        "score": 260,
    },
    {
        "title": "Addon Bootstrap Example",
        "question": "Show a minimal shared, server, and client addon skeleton in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            Files
            - lua/autorun/myaddon.lua
            - lua/autorun/server/sv_myaddon.lua
            - lua/autorun/client/cl_myaddon.lua
            - lua/myaddon/sh_config.lua

            lua/autorun/myaddon.lua
            ```lua
            if SERVER then
            AddCSLuaFile("autorun/client/cl_myaddon.lua")
            AddCSLuaFile("myaddon/sh_config.lua")
            include("myaddon/sh_config.lua")
            include("autorun/server/sv_myaddon.lua")
            return
            end

            include("myaddon/sh_config.lua")
            include("autorun/client/cl_myaddon.lua")
            ```

            lua/myaddon/sh_config.lua
            ```lua
            MYADDON = MYADDON or {}
            MYADDON.NetTag = "MyAddon_SyncState"
            ```

            lua/autorun/server/sv_myaddon.lua
            ```lua
            util.AddNetworkString(MYADDON.NetTag)

            hook.Add("PlayerInitialSpawn", "MyAddon_Welcome", function(ply)
            print("MyAddon loaded for " .. ply:Nick())
            end)
            ```

            lua/autorun/client/cl_myaddon.lua
            ```lua
            hook.Add("HUDPaint", "MyAddon_DebugText", function()
            draw.SimpleText("MyAddon loaded", "Trebuchet24", 20, 20, color_white)
            end)
            ```
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,hook",
        "score": 285,
    },
    {
        "title": "Server Client Realm Model",
        "question": "How do server and client work in Garry's Mod Lua, and what code belongs in each realm?",
        "answer": textwrap.dedent(
            """\
            Garry's Mod runs Lua in multiple realms.

            Server
            - Owns authoritative game state.
            - Spawns and removes entities.
            - Applies damage, validates input, and writes persistent save data.
            - Should be the source of truth for money, inventory, cooldowns, and permissions.

            Client
            - Draws HUD, VGUI, halos, particles, and model effects.
            - Handles LocalPlayer, camera code, and presentation.
            - Can request actions, but the server should validate them.

            Shared
            - Constants, shared utility functions, simple prediction-safe logic, and entity definitions.
            - Often used for SetupDataTables, shared method definitions, and network string names.

            Realm tools
            - Use if SERVER then and if CLIENT then for realm checks.
            - Use AddCSLuaFile on the server to send client and shared files.
            - Use net messages or networked vars to copy server state to clients.
            - Never trust the client for secure or persistent state.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook",
        "score": 255,
    },
    {
        "title": "Net Sync Example",
        "question": "Show an example of sending server-side addon data to the client in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            Server
            ```lua
            util.AddNetworkString("MyAddon_SyncCoins")

            hook.Add("PlayerInitialSpawn", "MyAddon_SendCoins", function(ply)
            local coins = 100
            net.Start("MyAddon_SyncCoins")
            net.WriteUInt(coins, 16)
            net.Send(ply)
            end)
            ```

            Client
            ```lua
            local myCoins = 0

            net.Receive("MyAddon_SyncCoins", function()
            myCoins = net.ReadUInt(16)
            end)

            hook.Add("HUDPaint", "MyAddon_DrawCoins", function()
            draw.SimpleText("Coins: " .. myCoins, "Trebuchet24", 20, 50, color_white)
            end)
            ```

            The server owns the real value. The client only receives a copy for display.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,hook",
        "score": 295,
    },
    {
        "title": "Players And Entities",
        "question": "What is the difference between players and entities in Garry's Mod Lua, and how should addon code handle them?",
        "answer": textwrap.dedent(
            """\
            Players are a specialized kind of Entity.

            Practical rules
            - Use Entity methods for generic world objects and shared entity behavior.
            - Use Player methods when you need player-only state such as SteamID64, Nick, Team, or inventory-like addon data.
            - Validate references with IsValid before using them.
            - If you need a player specifically, also check ent:IsPlayer().
            - Spawn, remove, parent, and mutate important state on the server.
            - Replicate only the state clients need to draw or predict.

            Common addon hooks
            - PlayerInitialSpawn and PlayerSpawn for player setup.
            - EntityTakeDamage and OnEntityCreated for world behavior.
            - ENT:Initialize, ENT:Think, ENT:Use, and ENT:OnRemove for scripted entities.
            - ENT:Draw and HUDPaint for client presentation.

            A good pattern is to keep server authority on the entity itself, then expose a small synchronized state to the owning player or nearby clients.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity,hook",
        "score": 265,
    },
    {
        "title": "Scripted Entity Ownership Example",
        "question": "Show an addon-style scripted entity that remembers its owner and can be used by players in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            lua/entities/sent_owner_crate/shared.lua
            ```lua
            ENT.Type = "anim"
            ENT.Base = "base_anim"
            ENT.PrintName = "Owner Crate"
            ENT.Spawnable = true

            function ENT:SetupDataTables()
            self:NetworkVar("Entity", 0, "OwnerPlayer")
            end
            ```

            lua/entities/sent_owner_crate/init.lua
            ```lua
            AddCSLuaFile("shared.lua")
            AddCSLuaFile("cl_init.lua")
            include("shared.lua")

            function ENT:Initialize()
            self:SetModel("models/props_junk/wood_crate001a.mdl")
            self:PhysicsInit(SOLID_VPHYSICS)
            self:SetMoveType(MOVETYPE_VPHYSICS)
            self:SetSolid(SOLID_VPHYSICS)
            local phys = self:GetPhysicsObject()
            if IsValid(phys) then
            phys:Wake()
            end
            end

            function ENT:Use(activator)
            if not IsValid(activator) or not activator:IsPlayer() then return end
            if not IsValid(self:GetOwnerPlayer()) then
            self:SetOwnerPlayer(activator)
            end
            activator:ChatPrint("Owner: " .. self:GetOwnerPlayer():Nick())
            end
            ```

            lua/entities/sent_owner_crate/cl_init.lua
            ```lua
            include("shared.lua")

            function ENT:Draw()
            self:DrawModel()
            end
            ```
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,entity,hook",
        "score": 305,
    },
    {
        "title": "Persistent Data Choices",
        "question": "What are the main ways to save persistent data in Garry's Mod, and when should I use each one?",
        "answer": textwrap.dedent(
            """\
            Garry's Mod addons usually save data in one of four ways.

            Player:SetPData and Player:GetPData
            - Simple per-player values stored with SQLite.
            - Good for small strings or numbers.
            - Convenient, but awkward for larger structured tables.

            file.Write and file.Read in garrysmod/data
            - Good for addon-wide JSON files or one file per player.
            - Works well with util.TableToJSON and util.JSONToTable.
            - Best when you want readable files or control over save format.

            sql.Query
            - Good for structured tables, filtering, and larger persistent systems.
            - Use sql.SQLStr when interpolating strings.

            cookie
            - Client-only preferences such as local UI state.
            - Not secure and not authoritative for gameplay systems.

            For important addon state, save on the server, key data by SteamID64 when possible, and sync only the pieces clients need.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity",
        "score": 270,
    },
    {
        "title": "Player Save Data Example",
        "question": "Show an example of saving and loading player addon data on the server in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            ```lua
            local SAVE_DIR = "myaddon"

            local function data_path(ply)
            return SAVE_DIR .. "/" .. ply:SteamID64() .. ".json"
            end

            local function load_player_data(ply)
            if not file.Exists(data_path(ply), "DATA") then
            return {coins = 0, loadout = {}}
            end

            local raw = file.Read(data_path(ply), "DATA") or ""
            local decoded = util.JSONToTable(raw) or {}
            decoded.coins = tonumber(decoded.coins) or 0
            decoded.loadout = istable(decoded.loadout) and decoded.loadout or {}
            return decoded
            end

            local function save_player_data(ply)
            if not IsValid(ply) or not ply.MyAddonData then return end
            file.CreateDir(SAVE_DIR)
            file.Write(data_path(ply), util.TableToJSON(ply.MyAddonData, true))
            end

            hook.Add("PlayerInitialSpawn", "MyAddon_LoadData", function(ply)
            ply.MyAddonData = load_player_data(ply)
            end)

            hook.Add("PlayerDisconnected", "MyAddon_SaveData", function(ply)
            save_player_data(ply)
            end)

            hook.Add("ShutDown", "MyAddon_SaveAllPlayers", function()
            for _, ply in ipairs(player.GetAll()) do
            save_player_data(ply)
            end
            end)
            ```

            This keeps persistence authoritative on the server and uses JSON files in garrysmod/data/myaddon.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,entity,hook",
        "score": 310,
    },
    {
        "title": "SWEP Realm Split",
        "question": "How is an addon-style SWEP usually split between shared, server, and client files in Garry's Mod Lua?",
        "answer": textwrap.dedent(
            """\
            Typical SWEP layout
            - shared.lua defines the weapon table, ammo, hold type, and shared behavior.
            - init.lua keeps server-only logic such as authoritative spawning, damage side effects, and AddCSLuaFile calls.
            - cl_init.lua keeps client-only drawing, crosshair, HUD, and presentation tweaks.

            Common pattern
            - Put SWEP metadata and SetupDataTables in shared.lua.
            - Keep gameplay authority on the server even if PrimaryAttack is predicted.
            - Put viewmodel, HUD, and client effects on the client.
            - Use net messages or networked vars when the client needs to display server-owned weapon state.

            This keeps weapon code compatible with addon-style file splitting while still matching Garry's Mod realm rules.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,swep,entity",
        "score": 275,
    },
]
EXPANDED_GMOD_MASTERY_CONCEPTS = [
    {
        "title": "GMod Debugging Workflow",
        "question": "How should I debug Garry's Mod Lua addons systematically?",
        "answer": textwrap.dedent(
            """\
            Use a layered debugging workflow instead of guessing.

            First checks
            - Confirm the realm: server, client, or shared.
            - Reproduce with the smallest hook, command, or entity possible.
            - Print the values you expect to change, not just markers like reached here.
            - Validate entities, players, and physics objects with IsValid.

            Useful tools
            - print, MsgN, MsgC, and PrintTable for fast inspection.
            - ErrorNoHalt when you want a non-fatal error path.
            - developer 1 plus debugoverlay or entity debug console commands for world inspection.
            - hook.GetTable() to inspect installed hooks.
            - timer.Exists and timer.Remove when debugging duplicate timers.

            Common causes
            - Running code in the wrong realm.
            - A hook name collision or duplicate hook.Add.
            - Invalid entities after cleanup, removal, or map changes.
            - Client expecting state that the server never sent.
            - Predicted code running more than once.

            A good addon workflow is: isolate the event, print the inputs, validate realm, validate entities, then only after that inspect networking or persistence.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook,entity",
        "score": 300,
    },
    {
        "title": "Hook Debugging Example",
        "question": "Show a simple way to debug a Garry's Mod hook without spamming meaningless output.",
        "answer": textwrap.dedent(
            """\
            ```lua
            local function dbg(tag, ...)
            MsgC(Color(0, 200, 255), "[MyAddon] ", color_white, tag .. ": ")
            print(...)
            end

            hook.Add("PlayerSpawn", "MyAddon_DebugSpawn", function(ply)
            if not IsValid(ply) then
            dbg("PlayerSpawn", "invalid player")
            return
            end

            dbg("PlayerSpawn", "realm=" .. (SERVER and "server" or "client"), ply:Nick(), ply:SteamID64())
            end)
            ```

            This gives each message a stable prefix, prints only the values that matter, and makes it obvious which hook and realm produced the output.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,hook,entity",
        "score": 315,
    },
    {
        "title": "Prediction Basics",
        "question": "How does prediction work in Garry's Mod Lua, and how do I avoid duplicated effects or logic?",
        "answer": textwrap.dedent(
            """\
            Prediction means some gameplay code runs on both client and server so controls feel responsive.

            Rules
            - Keep authoritative game state on the server.
            - Expect some weapon and movement code to run more than once.
            - Use IsFirstTimePredicted for client-only visual or audio side effects.
            - Do not duplicate server-only state mutations on the client.
            - Separate visual feedback from secure gameplay logic.

            Good uses for predicted client code
            - View kick, muzzle flash, local animation cues, and temporary UI feedback.

            Good uses for server code
            - Damage, ammo correction, cooldown authority, inventory changes, and persistence.

            If an effect appears twice or a sound plays twice, prediction is usually the first place to inspect.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook,swep",
        "score": 295,
    },
    {
        "title": "Predicted SWEP Attack Example",
        "question": "Show a prediction-safe SWEP attack pattern in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            ```lua
            function SWEP:PrimaryAttack()
            if not IsFirstTimePredicted() then return end

            self:SetNextPrimaryFire(CurTime() + 0.2)

            local owner = self:GetOwner()
            if not IsValid(owner) then return end

            owner:SetAnimation(PLAYER_ATTACK1)
            self:EmitSound("Weapon_Pistol.Single")

            if SERVER then
            local bullet = {}
            bullet.Num = 1
            bullet.Src = owner:GetShootPos()
            bullet.Dir = owner:GetAimVector()
            bullet.Spread = Vector(0.02, 0.02, 0)
            bullet.Damage = 12
            bullet.Attacker = owner
            owner:FireBullets(bullet)
            self:TakePrimaryAmmo(1)
            end
            end
            ```

            The client gets responsive feedback immediately, but damage and ammo remain server-authoritative.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,swep,entity",
        "score": 325,
    },
    {
        "title": "Gamemode Architecture",
        "question": "How should I structure a larger Garry's Mod gamemode or complex addon system?",
        "answer": textwrap.dedent(
            """\
            Treat a gamemode or large addon like a set of subsystems instead of one huge file.

            Useful layers
            - Bootstrapping: file loading and realm setup.
            - Config: constants, enums, tunables, permissions.
            - Domain systems: inventory, jobs, economy, quests, crafting, housing, or combat.
            - Entities: scripted entities, weapons, effects, NPC helpers.
            - Transport: net messages and state synchronization.
            - Persistence: save/load, migrations, and shutdown handling.
            - UI: menus, HUD, context panels, notifications.

            Keep each subsystem narrow. A good rule is that UI should ask systems for state, systems should ask persistence layers to save, and only the server should be allowed to mutate the authoritative state.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook,entity",
        "score": 290,
    },
    {
        "title": "Inventory System Design",
        "question": "How do I design a complex inventory or economy system in Garry's Mod Lua?",
        "answer": textwrap.dedent(
            """\
            Use a server-owned state model.

            Recommended structure
            - Store player state in a plain Lua table keyed by SteamID64.
            - Use item definitions for static metadata and a separate inventory table for per-player state.
            - Validate every purchase, item use, and transfer on the server.
            - Save the authoritative state periodically and on disconnect or shutdown.
            - Send compact snapshots or diffs to clients for UI.

            Avoid
            - Storing the real inventory only on the client.
            - Trusting net messages to contain valid prices or item counts.
            - Mixing UI code directly into persistence logic.

            This same pattern works for currencies, skill trees, housing ownership, loadouts, quest progress, and crafting systems.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity,hook",
        "score": 305,
    },
    {
        "title": "Inventory Sync Example",
        "question": "Show a simple Garry's Mod example of a server-owned inventory syncing to a client UI.",
        "answer": textwrap.dedent(
            """\
            Server
            ```lua
            util.AddNetworkString("MyAddon_InventorySync")

            local function sync_inventory(ply)
            if not IsValid(ply) then return end
            net.Start("MyAddon_InventorySync")
            net.WriteString(util.TableToJSON(ply.MyInventory or {}, false) or "[]")
            net.Send(ply)
            end

            hook.Add("PlayerInitialSpawn", "MyAddon_InitInventory", function(ply)
            ply.MyInventory = {{id = "medkit", amount = 2}, {id = "battery", amount = 1}}
            sync_inventory(ply)
            end)
            ```

            Client
            ```lua
            local inventory = {}

            net.Receive("MyAddon_InventorySync", function()
            inventory = util.JSONToTable(net.ReadString()) or {}
            end)
            ```

            The server owns the inventory. The client only displays a synchronized copy.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,entity,hook",
        "score": 330,
    },
    {
        "title": "VGUI Architecture",
        "question": "How should I build larger VGUI interfaces in Garry's Mod without making them messy?",
        "answer": textwrap.dedent(
            """\
            Treat VGUI as a view layer.

            Good rules
            - Keep panel creation separate from data fetching and validation.
            - Build reusable panel classes instead of one giant frame function.
            - Have the UI render client-side copies of state, not the authoritative server state itself.
            - Use hooks or explicit refresh functions when network data changes.
            - Keep layout, styling, and interaction callbacks readable and localized.

            For large menus, split panels into purpose-built widgets such as item rows, category tabs, stat cards, and confirmation dialogs. Server logic should never depend on a button existing on the client.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon",
        "score": 285,
    },
    {
        "title": "Trace Interaction Example",
        "question": "Show a Garry's Mod trace-based interaction example for players and entities.",
        "answer": textwrap.dedent(
            """\
            ```lua
            concommand.Add("myaddon_use_trace", function(ply)
            if not IsValid(ply) then return end

            local trace = util.TraceLine({
            start = ply:EyePos(),
            endpos = ply:EyePos() + ply:GetAimVector() * 120,
            filter = ply,
            })

            local ent = trace.Entity
            if not IsValid(ent) then
            ply:ChatPrint("No valid entity hit")
            return
            end

            ply:ChatPrint("You hit: " .. tostring(ent))
            end)
            ```

            Traces are a core building block for interaction systems, context use, construction tools, abilities, and target selection.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,entity",
        "score": 315,
    },
    {
        "title": "Rendering And Effects Architecture",
        "question": "How should I think about client rendering, 3D2D, and visual effects in Garry's Mod Lua?",
        "answer": textwrap.dedent(
            """\
            Rendering belongs on the client.

            Common responsibilities
            - HUDPaint for 2D screen overlays.
            - PostDrawOpaqueRenderables or ENT:Draw for world rendering.
            - cam.Start3D2D and cam.End3D2D for world-space panels or labels.
            - render and surface libraries for materials, beams, sprites, and simple shapes.

            Good pattern
            - The server sends compact state.
            - The client turns that state into markers, outlines, 3D2D labels, or HUD widgets.
            - Expensive drawing is gated by distance, visibility, or relevance.

            Do not make the client authoritative for gameplay just because it draws the effect.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity",
        "score": 295,
    },
    {
        "title": "Performance Checklist",
        "question": "What are the main performance rules for Garry's Mod Lua addons and complex systems?",
        "answer": textwrap.dedent(
            """\
            Performance problems usually come from code that runs too often or sends too much.

            Checklist
            - Do not do heavy work in Think or HUDPaint unless it is cached or gated.
            - Cache lookups, panels, and expensive calculations when possible.
            - Prefer event-driven hooks over polling loops.
            - Avoid broadcasting net messages when only one player needs the data.
            - Avoid serializing giant tables every frame.
            - Reuse timers and materials instead of recreating them repeatedly.
            - Be careful with ents.GetAll, player.GetAll, and large table scans in hot paths.
            - Cull expensive drawing by distance, visibility, or ownership.

            When debugging lag, first identify the hot hook, then reduce frequency, allocations, and network volume.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook,entity",
        "score": 310,
    },
    {
        "title": "Net Security Checklist",
        "question": "How do I secure Garry's Mod net messages, commands, and complex addon systems against abuse?",
        "answer": textwrap.dedent(
            """\
            Security rules
            - Never trust the client to tell you what is allowed.
            - Validate entity references, ownership, range, and permissions on the server.
            - Validate numbers, strings, and table contents from net messages.
            - Add cooldowns or rate limits for expensive actions.
            - Re-check money, inventory, job, admin, or quest requirements server-side.
            - Ignore or punish malformed requests instead of trying to use them.

            Dangerous pattern
            - Client sends damage, item count, price, or admin-only intent and server blindly applies it.

            Safe pattern
            - Client sends a request identifier.
            - Server looks up the real state and decides whether the action is legal.
            - Server mutates the authoritative state and syncs the result back.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,hook,entity",
        "score": 320,
    },
    {
        "title": "NextBot And NPC Design",
        "question": "How should I approach AI, NextBots, and NPC behavior systems in Garry's Mod Lua?",
        "answer": textwrap.dedent(
            """\
            Keep AI server-side and model it as states.

            Good pattern
            - Sense: gather targets, distances, visibility, and navigation status.
            - Decide: choose idle, patrol, chase, attack, flee, or interact.
            - Act: move, play animations, trigger attacks, or change schedule.

            Tips
            - Use simple state machines before trying advanced planners.
            - Separate navigation, combat, and presentation.
            - Keep expensive searches throttled with timers or intervals.
            - Replicate only the client visuals you need, such as effects or overlays.

            This pattern also works for turret logic, RTS units, companion NPCs, and enemy waves.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity,hook",
        "score": 295,
    },
    {
        "title": "HTTP JSON Example",
        "question": "Show a safe pattern for fetching JSON data in a Garry's Mod addon.",
        "answer": textwrap.dedent(
            """\
            ```lua
            http.Fetch("https://example.com/myaddon/config.json",
            function(body)
            local data = util.JSONToTable(body)
            if not istable(data) then
            ErrorNoHalt("[MyAddon] Invalid JSON response\n")
            return
            end

            print("Fetched config version:", data.version)
            end,
            function(err)
            ErrorNoHalt("[MyAddon] HTTP failed: " .. tostring(err) .. "\n")
            end)
            ```

            Treat remote data as untrusted input. Validate shape and types before you use it in gameplay code.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon",
        "score": 305,
    },
    {
        "title": "Creative Addon Ideas",
        "question": "What are some good advanced addon ideas for Garry's Mod Lua that combine systems creatively?",
        "answer": textwrap.dedent(
            """\
            Strong GMod projects usually combine several small systems into one loop.

            Examples
            - A salvage system: trace-based world interaction, entity ownership, inventory, crafting, and persistence.
            - A dynamic bounty system: player data, score tracking, HUD markers, net sync, and rewards.
            - A base defense addon: scripted entities, power networks, resource generation, NPC waves, and save data.
            - A detective toolset: traces, 3D2D markers, client HUD, evidence entities, and per-round cleanup.
            - A vehicle progression system: vehicle entities, upgrades, persistence, and garage UI.

            The best ideas are not just one mechanic. They combine hooks, entities, VGUI, networking, and persistence into a coherent loop.
            """
        ).strip(),
        "kind": "guide-reference",
        "focus_tags": "addon,entity,hook,swep",
        "score": 300,
    },
    {
        "title": "Validated Net Receive Example",
        "question": "Show a safe server-side net.Receive pattern for a Garry's Mod addon action.",
        "answer": textwrap.dedent(
            """\
            ```lua
            util.AddNetworkString("MyAddon_RequestUse")

            net.Receive("MyAddon_RequestUse", function(_, ply)
            if not IsValid(ply) then return end

            local ent = net.ReadEntity()
            if not IsValid(ent) or ent:GetPos():DistToSqr(ply:GetPos()) > (128 * 128) then
            return
            end

            if ent.UseCooldown and ent.UseCooldown > CurTime() then return end
            ent.UseCooldown = CurTime() + 0.25

            if ent.MyAddonUse then
            ent:MyAddonUse(ply)
            end
            end)
            ```

            The server validates the player, entity, distance, and cooldown before doing any addon action. Never trust the client request by itself.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,hook,entity",
        "score": 345,
    },
    {
        "title": "Player Spawn Loadout Example",
        "question": "Show a clean Garry's Mod PlayerSpawn hook that gives a loadout on the server.",
        "answer": textwrap.dedent(
            """\
            ```lua
            hook.Add("PlayerSpawn", "MyAddon_GiveLoadout", function(ply)
            if not IsValid(ply) then return end

            ply:StripWeapons()
            ply:Give("weapon_crowbar")
            ply:Give("weapon_pistol")
            ply:GiveAmmo(60, "Pistol", true)
            ply:SetHealth(100)
            ply:SetArmor(25)
            end)
            ```

            Keep loadout logic on the server so weapons, ammo, and stats stay authoritative.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,hook,swep",
        "score": 338,
    },
    {
        "title": "Minimal Scripted Entity Skeleton",
        "question": "Show a minimal addon-style scripted entity skeleton in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            Files
            - lua/entities/sent_myaddon/shared.lua
            - lua/entities/sent_myaddon/init.lua
            - lua/entities/sent_myaddon/cl_init.lua

            shared.lua
            ```lua
            ENT.Type = "anim"
            ENT.Base = "base_anim"
            ENT.PrintName = "My Addon Entity"
            ENT.Spawnable = true

            function ENT:SetupDataTables()
            self:NetworkVar("Bool", 0, "Enabled")
            end
            ```

            init.lua
            ```lua
            AddCSLuaFile("shared.lua")
            AddCSLuaFile("cl_init.lua")
            include("shared.lua")

            function ENT:Initialize()
            self:SetModel("models/props_c17/oildrum001.mdl")
            self:PhysicsInit(SOLID_VPHYSICS)
            self:SetMoveType(MOVETYPE_VPHYSICS)
            self:SetSolid(SOLID_VPHYSICS)
            self:SetEnabled(true)
            end
            ```

            cl_init.lua
            ```lua
            include("shared.lua")

            function ENT:Draw()
            self:DrawModel()
            end
            ```

            This is the standard addon-style entity split: shared metadata, server initialization in init.lua, and client drawing code in cl_init.lua.
            """ 
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,entity,hook",
        "score": 350,
    },
    {
        "title": "Minimal SWEP Skeleton",
        "question": "Show a minimal addon-style SWEP skeleton in Garry's Mod Lua.",
        "answer": textwrap.dedent(
            """\
            Files
            - lua/weapons/weapon_myaddon/shared.lua
            - lua/weapons/weapon_myaddon/init.lua
            - lua/weapons/weapon_myaddon/cl_init.lua

            shared.lua
            ```lua
            SWEP.PrintName = "My Addon Weapon"
            SWEP.Author = "MyAddon"
            SWEP.Spawnable = true
            SWEP.UseHands = true
            SWEP.ViewModel = "models/weapons/c_pistol.mdl"
            SWEP.WorldModel = "models/weapons/w_pistol.mdl"
            SWEP.Primary.ClipSize = 12
            SWEP.Primary.DefaultClip = 12
            SWEP.Primary.Automatic = false
            SWEP.Primary.Ammo = "Pistol"

            function SWEP:PrimaryAttack()
            if not IsFirstTimePredicted() then return end
            self:SetNextPrimaryFire(CurTime() + 0.25)
            end
            ```

            init.lua
            ```lua
            AddCSLuaFile("shared.lua")
            AddCSLuaFile("cl_init.lua")
            include("shared.lua")
            ```

            cl_init.lua
            ```lua
            include("shared.lua")
            ```

            This is the normal addon-style skeleton: shared definitions first, server bootstrap in init.lua, and client include logic in cl_init.lua.
            """
        ).strip(),
        "kind": "guide-example",
        "focus_tags": "addon,swep,entity",
        "score": 342,
    },
]
LUA_PRIORITY_KEYWORDS = {
    "lexical": 12,
    "values and types": 12,
    "variables": 12,
    "statements": 18,
    "expressions": 18,
    "table": 18,
    "function": 24,
    "visibility": 10,
    "error": 14,
    "metatable": 18,
    "environment": 16,
    "coroutine": 12,
    "module": 16,
    "string": 12,
    "basic functions": 18,
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def safe_int(value, default=0):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def load_uploaded_training_rows():
    if not UPLOADED_FILES_TRAIN_PATH.exists():
        return []

    rows = []
    with UPLOADED_FILES_TRAIN_PATH.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue

            question = normalize_text(payload.get("question") or "")
            answer = normalize_text(payload.get("answer") or "")
            if not question or not answer:
                continue

            source = normalize_text(payload.get("source") or f"upload:{line_number}")
            title = normalize_text(payload.get("title") or pathlib.Path(str(payload.get("path") or source)).name or f"upload-{line_number}")
            kind = normalize_text(payload.get("kind") or "user-upload-example")
            focus_tags = normalize_text(payload.get("focus_tags") or "addon")
            score = safe_int(payload.get("score"), 320)

            rows.append({
                "question": question,
                "answer": answer,
                "prompt": normalize_text(payload.get("prompt") or question),
                "completion": normalize_text(payload.get("completion") or answer),
                "source": source,
                "dataset": normalize_text(payload.get("dataset") or "upload"),
                "title": title,
                "kind": kind,
                "focus_tags": focus_tags,
                "score": score,
                "page_score": safe_int(payload.get("page_score"), score),
            })

    return deduplicate_rows(rows)


def iter_scraped_datasets():
    if not SCRAPED_DATA_DIR.exists():
        return []
    return sorted(path for path in SCRAPED_DATA_DIR.glob("*.json") if path.is_file())


def is_lua_manual_dataset(data):
    source = normalize_text(data.get("source") or data.get("siteUrl") or "").lower()
    site_name = normalize_text(data.get("siteName") or "").lower()
    if "lua.org/manual/5.1" in source:
        return True
    if site_name in {"5.1", "lua51", "manual"}:
        return True
    for page in data.get("pages") or []:
        if str(page.get("address") or "").strip().lower() == "manual.html":
            return True
    return False


def load_gmod_source_datasets():
    datasets = []
    if LEGACY_GMOD_SOURCE.exists():
        datasets.append((LEGACY_GMOD_SOURCE, read_json(LEGACY_GMOD_SOURCE)))
    for path in iter_scraped_datasets():
        data = read_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
            continue
        if is_lua_manual_dataset(data):
            continue
        datasets.append((path, data))
    return datasets


def load_lua_source_dataset():
    if LEGACY_LUA_SOURCE.exists():
        return LEGACY_LUA_SOURCE, read_json(LEGACY_LUA_SOURCE)
    for path in iter_scraped_datasets():
        data = read_json(path)
        if not isinstance(data, dict) or not isinstance(data.get("pages"), list):
            continue
        if is_lua_manual_dataset(data):
            return path, data
    return None, None


def normalize_text(text):
    if not text:
        return ""
    text = html.unescape(str(text))
    text = text.replace("\r", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_html(source):
    if not source:
        return ""
    text = source
    text = re.sub(r"(?i)<br\s*/?>", "\n", text)
    text = re.sub(r"(?i)</p>", "\n\n", text)
    text = re.sub(r"(?i)<p[^>]*>", "\n\n", text)
    text = re.sub(r"(?i)<li[^>]*>", "\n- ", text)
    text = re.sub(r"(?i)</li>", "", text)
    text = re.sub(r"(?i)<pre[^>]*>", "\n```lua\n", text)
    text = re.sub(r"(?i)</pre>", "\n```\n", text)
    text = re.sub(r"(?i)<code[^>]*>", "`", text)
    text = re.sub(r"(?i)</code>", "`", text)
    text = re.sub(r"(?is)<[^>]+>", " ", text)
    return normalize_text(text)


def trim_text(text, limit=3500):
    text = normalize_text(text)
    if len(text) <= limit:
        return text
    cut = text[:limit]
    split_at = max(cut.rfind("\n\n"), cut.rfind(". "), cut.rfind("\n"))
    if split_at >= limit // 2:
        cut = cut[:split_at]
    return cut.strip() + "\n\n[Truncated]"


def format_list_block(items, heading, formatter):
    if not items:
        return ""
    lines = [heading]
    for item in items:
        formatted = formatter(item)
        if formatted:
            lines.append(formatted)
    return "\n".join(lines).strip()


def format_argument(item):
    name = normalize_text(item.get("name") or "value")
    arg_type = normalize_text(item.get("type") or "any")
    description = normalize_text(item.get("description") or "")
    line = f"- {name}: {arg_type}"
    if description:
        line += f". {description}"
    return line


def format_return(item):
    name = normalize_text(item.get("name") or "value")
    return_type = normalize_text(item.get("type") or "any")
    description = normalize_text(item.get("description") or "")
    line = f"- {name}: {return_type}"
    if description:
        line += f". {description}"
    return line


def format_notes(items, heading):
    if not items:
        return ""
    cleaned = []
    for item in items:
        if isinstance(item, dict):
            value = item.get("description") or item.get("text") or json.dumps(item, ensure_ascii=False)
        else:
            value = item
        value = normalize_text(value)
        if value:
            cleaned.append(f"- {value}")
    if not cleaned:
        return ""
    return "\n".join([heading] + cleaned)


def select_example(page):
    examples = page.get("examples") or []
    if not examples:
        return ""
    chosen = []
    for example in examples[:2]:
        description = normalize_text(example.get("description") or "")
        code = normalize_text(example.get("code") or "")
        parts = []
        if description:
            parts.append(description)
        if code:
            language = normalize_text(example.get("language") or "lua")
            parts.append(f"```{language}\n{code}\n```")
        if parts:
            chosen.append("\n".join(parts))
    if not chosen:
        return ""
    return "\n\n".join(["Example"] + chosen)


def build_example_blob(page):
    parts = []
    for example in page.get("examples") or []:
        if not isinstance(example, dict):
            continue
        description = normalize_text(example.get("description") or "")
        code = normalize_text(example.get("code") or "")
        if description:
            parts.append(description)
        if code:
            parts.append(code)
    return "\n".join(parts)


def classify_gmod_focus(page):
    raw_title = str(page.get("title") or page.get("address") or "")
    title = normalize_text(raw_title)
    address = normalize_text(page.get("address") or "")
    parent = normalize_text(page.get("parent") or "")
    member_type = normalize_text(page.get("memberType") or "").lower()
    example_blob = build_example_blob(page)

    tags = set()
    if raw_title.startswith(SWEP_PREFIXES) or parent.upper() in {"WEAPON", "SWEP"}:
        tags.add("swep")
    if member_type in {"hook", "panelhook"} or raw_title.startswith(HOOK_TITLE_PREFIXES):
        tags.add("hook")
    if raw_title.startswith(ENTITY_METHOD_PREFIXES) or raw_title.startswith(ENTITY_HOOK_PREFIXES) or address.startswith("ents.") or parent in {"Entity", "Player", "Weapon", "NPC", "NextBot", "Vehicle"}:
        tags.add("entity")
    if raw_title.startswith(PANEL_PREFIXES):
        tags.add("addon")
    if title in CORE_ADDON_API_TITLES or address in CORE_ADDON_API_TITLES or title.startswith("Tutorials -") or ADDON_CODE_PATTERN.search(example_blob):
        tags.add("addon")
    return sorted(tags)


def score_gmod_page(page, focus_tags):
    title = normalize_text(page.get("title") or page.get("address") or "")
    address = normalize_text(page.get("address") or "")
    member_type = normalize_text(page.get("memberType") or "").lower()
    page_type = normalize_text(page.get("pageType") or "").lower()
    examples = len(page.get("examples") or [])
    score = 0

    score += min(examples, 5) * 6
    score += min(int(page.get("codeBlockCount") or 0), 4) * 4
    if page.get("arguments"):
        score += 12
    if page.get("returns"):
        score += 10
    if page.get("realms"):
        score += 6
    if page.get("description"):
        score += 6
    if member_type in {"hook", "panelhook"}:
        score += 18
    elif member_type in {"classfunc", "libraryfunc", "panelfunc"}:
        score += 8
    if page_type == "function":
        score += 8
    if "swep" in focus_tags:
        score += 65
    if "hook" in focus_tags:
        score += 55
    if "entity" in focus_tags:
        score += 48
    if "addon" in focus_tags:
        score += 24
    if title in CORE_ADDON_API_TITLES or address in CORE_ADDON_API_TITLES:
        score += 35
    return score


def score_gmod_row(page_score, row_kind, focus_tags):
    score = page_score
    if row_kind.endswith("-example"):
        score += 42
    elif row_kind.endswith("-signature"):
        score += 10
    else:
        score += 4
    if "addon" in focus_tags and row_kind.endswith("-example"):
        score += 10
    return score


def build_gmod_reference_answer(page):
    sections = []
    description = normalize_text(page.get("description") or "")
    if description:
        sections.append(description)

    realms = page.get("realms") or []
    realms = [normalize_text(value) for value in realms if normalize_text(value)]
    if realms:
        sections.append("Realms\n- " + "\n- ".join(realms))

    arguments = format_list_block(page.get("arguments") or [], "Arguments", format_argument)
    if arguments:
        sections.append(arguments)

    returns = format_list_block(page.get("returns") or [], "Returns", format_return)
    if returns:
        sections.append(returns)

    notes = format_notes(page.get("notes") or [], "Notes")
    if notes:
        sections.append(notes)

    warnings = format_notes(page.get("warnings") or [], "Warnings")
    if warnings:
        sections.append(warnings)

    deprecated = format_notes(page.get("deprecated") or [], "Deprecated")
    if deprecated:
        sections.append(deprecated)

    example = select_example(page)
    if example:
        sections.append(example)

    fallback = normalize_text(page.get("textContent") or "")
    combined = "\n\n".join(sections).strip()
    if fallback and (not sections or len(combined) < 180):
        sections.append(trim_text(fallback, limit=2500))

    if not sections:
        sections.append("No documentation available.")

    return trim_text("\n\n".join(sections), limit=4000)


def build_gmod_reference_question(page):
    title = normalize_text(page.get("title") or page.get("address") or "Unknown")
    page_type = normalize_text(page.get("pageType") or "page").lower()
    if page_type == "function":
        return f"In Garry's Mod Lua, what does {title} do?"
    if page_type in {"cat", "type", "panel", "structure", "enum", "title"}:
        return f"Explain {title} in Garry's Mod Lua."
    return f"Explain {title} in Garry's Mod Lua."


def build_signature_answer(page):
    sections = []
    arguments = format_list_block(page.get("arguments") or [], "Arguments", format_argument)
    returns = format_list_block(page.get("returns") or [], "Returns", format_return)
    realms = page.get("realms") or []

    if realms:
        sections.append("Realms\n- " + "\n- ".join(normalize_text(value) for value in realms if normalize_text(value)))
    if arguments:
        sections.append(arguments)
    if returns:
        sections.append(returns)

    if not sections:
        description = normalize_text(page.get("description") or "")
        sections.append(description or "No argument or return documentation is available.")

    return trim_text("\n\n".join(sections), limit=3000)


def build_example_answer(page):
    example = select_example(page)
    if example:
        return trim_text(example, limit=3000)
    return build_gmod_reference_answer(page)


def build_gmod_rows():
    rows = []
    source_datasets = load_gmod_source_datasets()
    if not source_datasets:
        rows.extend(build_curated_gmod_concept_rows())
        return rows

    skip_page_types = {"ambig", "warning"}
    seen_pages = set()
    for _, data in source_datasets:
        for page in data["pages"]:
            if not page.get("isCodingPage"):
                continue
            if (page.get("pageType") or "").lower() in skip_page_types:
                continue

            page_key = normalize_text(page.get("url") or page.get("address") or page.get("title") or "")
            if page_key and page_key in seen_pages:
                continue
            if page_key:
                seen_pages.add(page_key)

            title = normalize_text(page.get("title") or page.get("address") or "Unknown")
            page_type = normalize_text(page.get("pageType") or "page")
            member_type = normalize_text(page.get("memberType") or "")
            kind = member_type or page_type
            source = page.get("url") or f"gmod:{page.get('address') or title}"
            focus_tags = classify_gmod_focus(page)
            focus_tag_text = ",".join(focus_tags)
            page_score = score_gmod_page(page, focus_tags)

            reference_answer = build_gmod_reference_answer(page)
            reference_question = build_gmod_reference_question(page)
            reference_kind = kind
            rows.append({
                "question": reference_question,
                "answer": reference_answer,
                "prompt": reference_question,
                "completion": reference_answer,
                "source": source,
                "dataset": "gmod",
                "title": title,
                "kind": reference_kind,
                "focus_tags": focus_tag_text,
                "score": score_gmod_row(page_score, reference_kind, focus_tags),
                "page_score": page_score,
            })

            if page.get("arguments") or page.get("returns") or page.get("realms"):
                signature_answer = build_signature_answer(page)
                signature_kind = f"{kind}-signature"
                rows.append({
                    "question": f"What are the arguments, return values, and realm details for {title} in Garry's Mod Lua?",
                    "answer": signature_answer,
                    "prompt": f"What are the arguments, return values, and realm details for {title} in Garry's Mod Lua?",
                    "completion": signature_answer,
                    "source": source,
                    "dataset": "gmod",
                    "title": title,
                    "kind": signature_kind,
                    "focus_tags": focus_tag_text,
                    "score": score_gmod_row(page_score, signature_kind, focus_tags),
                    "page_score": page_score,
                })

            if page.get("examples"):
                example_answer = build_example_answer(page)
                example_kind = f"{kind}-example"
                rows.append({
                    "question": f"Show an example of how to use {title} in Garry's Mod Lua.",
                    "answer": example_answer,
                    "prompt": f"Show an example of how to use {title} in Garry's Mod Lua.",
                    "completion": example_answer,
                    "source": source,
                    "dataset": "gmod",
                    "title": title,
                    "kind": example_kind,
                    "focus_tags": focus_tag_text,
                    "score": score_gmod_row(page_score, example_kind, focus_tags),
                    "page_score": page_score,
                })
    rows.extend(build_curated_gmod_concept_rows())
    return rows


def build_curated_gmod_concept_rows():
    rows = []
    for item in CURATED_GMOD_CONCEPTS + EXPANDED_GMOD_MASTERY_CONCEPTS:
        title = normalize_text(item["title"])
        question = normalize_text(item["question"])
        answer = trim_text(item["answer"], limit=4000)
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        rows.append({
            "question": question,
            "answer": answer,
            "prompt": question,
            "completion": answer,
            "source": f"curated:gmod:{slug}",
            "dataset": "gmod",
            "title": title,
            "kind": item["kind"],
            "focus_tags": item["focus_tags"],
            "score": item["score"],
            "page_score": item["score"],
        })
    return rows


def split_large_section(heading, text, limit=2800):
    text = normalize_text(text)
    if len(text) <= limit:
        return [(heading, text)]

    paragraphs = [part.strip() for part in text.split("\n\n") if part.strip()]
    chunks = []
    current = []
    current_length = 0
    for paragraph in paragraphs:
        addition = len(paragraph) + (2 if current else 0)
        if current and current_length + addition > limit:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_length = len(paragraph)
        else:
            current.append(paragraph)
            current_length += addition
    if current:
        chunks.append("\n\n".join(current))

    if len(chunks) <= 1:
        return [(heading, trim_text(text, limit=limit))]

    results = []
    total = len(chunks)
    for index, chunk in enumerate(chunks, start=1):
        results.append((f"{heading} (part {index}/{total})", chunk))
    return results


def build_lua_manual_rows():
    source_path, data = load_lua_source_dataset()
    if not source_path or not data:
        return []

    manual_page = next(page for page in data["pages"] if page.get("address") == "manual.html")
    raw_html = manual_page.get("rawHtml") or ""
    heading_pattern = re.compile(r"<H([23])[^>]*>(.*?)</H\1>", re.I | re.S)
    matches = list(heading_pattern.finditer(raw_html))
    rows = []

    for index, match in enumerate(matches):
        heading_text = strip_html(match.group(2))
        if not heading_text:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else raw_html.find('<P CLASS="footer"')
        if end == -1:
            end = len(raw_html)
        section_html = raw_html[start:end]
        section_text = strip_html(section_html)
        section_text = re.sub(r"^contents\s+·\s+index\s+·\s+other versions.*?\n", "", section_text, flags=re.I)
        if len(section_text) < 80:
            continue

        for split_heading, split_text in split_large_section(heading_text, section_text):
            answer = trim_text(split_text, limit=3200)
            row_score = score_lua_row(split_heading)
            rows.append({
                "question": f"Explain Lua 5.1 section {split_heading}.",
                "answer": answer,
                "prompt": f"Explain Lua 5.1 section {split_heading}.",
                "completion": answer,
                "source": f"https://www.lua.org/manual/5.1/manual.html#{normalize_text(heading_text)}",
                "dataset": "lua5.1",
                "title": split_heading,
                "kind": "manual-section",
                "focus_tags": "lua-foundation",
                "score": row_score,
                "page_score": row_score,
            })
    return rows


def score_lua_row(title):
    lowered = normalize_text(title).lower()
    score = 0
    for keyword, points in LUA_PRIORITY_KEYWORDS.items():
        if keyword in lowered:
            score += points
    if lowered.startswith("2."):
        score += 8
    if lowered.startswith("5."):
        score += 6
    if score == 0:
        score = 4
    return score


def sort_rows(rows):
    return sorted(rows, key=lambda row: (row.get("score", 0), row.get("page_score", 0), row.get("title", ""), row.get("kind", "")), reverse=True)


def select_top_diverse_rows(rows, target_count, per_title_cap=2, min_score=0):
    selected = []
    title_counts = {}
    for row in sort_rows(rows):
        if row.get("score", 0) < min_score:
            continue
        title = row.get("title", "")
        if title_counts.get(title, 0) >= per_title_cap:
            continue
        selected.append(row)
        title_counts[title] = title_counts.get(title, 0) + 1
        if len(selected) >= target_count:
            break
    return selected


def deduplicate_rows(rows):
    unique_rows = []
    seen = set()
    for row in rows:
        key = (
            row.get("question", ""),
            row.get("answer", ""),
            row.get("source", ""),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def build_coding_dataset(gmod_rows, lua_rows):
    coding_foundation_titles = {
        "Addon Structure And Realms",
        "Server Client Realm Model",
        "Prediction Basics",
        "SWEP Realm Split",
    }
    coding_example_rows = select_top_diverse_rows(
        [
            row for row in gmod_rows
            if row.get("kind", "").endswith("-example")
            and any(tag in row.get("focus_tags", "") for tag in ("addon", "hook", "entity", "swep"))
        ],
        target_count=1800,
        per_title_cap=2,
        min_score=110,
    )
    coding_signature_rows = select_top_diverse_rows(
        [
            row for row in gmod_rows
            if row.get("kind", "").endswith("-signature")
            and any(tag in row.get("focus_tags", "") for tag in ("addon", "hook", "entity", "swep"))
        ],
        target_count=900,
        per_title_cap=2,
        min_score=95,
    )
    coding_foundation_rows = select_top_diverse_rows(
        [
            row for row in gmod_rows
            if row.get("title") in coding_foundation_titles
        ],
        target_count=40,
        per_title_cap=1,
        min_score=0,
    )
    coding_lua_rows = select_top_diverse_rows(lua_rows, target_count=40, per_title_cap=1, min_score=18)
    coding_gmod_rows = deduplicate_rows(coding_example_rows + coding_signature_rows + coding_foundation_rows)
    return coding_gmod_rows + coding_lua_rows


def build_focused_datasets(gmod_rows, lua_rows):
    focused_gmod_rows = [
        row for row in gmod_rows
        if row.get("focus_tags") or row.get("score", 0) >= 95
    ]
    focused_lua_rows = select_top_diverse_rows(lua_rows, target_count=80, per_title_cap=1, min_score=14)
    focused_rows = sort_rows(focused_gmod_rows) + focused_lua_rows

    quickstart_focus_rows = [row for row in focused_gmod_rows if row.get("focus_tags")]
    quickstart_example_rows = select_top_diverse_rows(
        [row for row in quickstart_focus_rows if row["kind"].endswith("-example")],
        target_count=1300,
        per_title_cap=1,
        min_score=110,
    )
    quickstart_reference_rows = select_top_diverse_rows(
        [row for row in quickstart_focus_rows if not row["kind"].endswith("-example")],
        target_count=900,
        per_title_cap=1,
        min_score=105,
    )
    quickstart_gmod_rows = sort_rows(quickstart_example_rows + quickstart_reference_rows)
    quickstart_lua_rows = select_top_diverse_rows(lua_rows, target_count=50, per_title_cap=1, min_score=18)
    quickstart_rows = quickstart_gmod_rows + quickstart_lua_rows
    return focused_rows, quickstart_rows


def write_csv(path, rows, columns):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_messages_jsonl(path, rows):
    message_rows = []
    for row in rows:
        message_rows.append({
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]},
            ],
            "source": row["source"],
            "dataset": row["dataset"],
            "title": row["title"],
            "kind": row["kind"],
            "focus_tags": row.get("focus_tags", ""),
            "score": row.get("score", 0),
        })
    write_jsonl(path, message_rows)


def remove_legacy_unsloth_outputs():
    legacy_paths = [
        DATASETS_DIR / "gmod_reference_unsloth.jsonl",
        DATASETS_DIR / "gmod_reference_unsloth.csv",
        DATASETS_DIR / "lua51_reference_unsloth.jsonl",
        DATASETS_DIR / "lua51_reference_unsloth.csv",
        DATASETS_DIR / "gmod_lua_unsloth_train.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_train.csv",
        DATASETS_DIR / "gmod_lua_unsloth_eval.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_eval.csv",
        DATASETS_DIR / "gmod_lua_unsloth_focused_train.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_focused_train.csv",
        DATASETS_DIR / "gmod_lua_unsloth_focused_eval.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_focused_eval.csv",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_train.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_train.csv",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_eval.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_eval.csv",
        DATASETS_DIR / "gmod_lua_unsloth_train_messages.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_eval_messages.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_focused_train_messages.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_focused_eval_messages.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_train_messages.jsonl",
        DATASETS_DIR / "gmod_lua_unsloth_quickstart_eval_messages.jsonl",
    ]
    for path in legacy_paths:
        if path.exists():
            path.unlink()


def split_rows(rows, eval_ratio=0.02, min_eval=100, max_eval=400):
    shuffled = list(rows)
    random.Random(RANDOM_SEED).shuffle(shuffled)
    eval_size = int(len(shuffled) * eval_ratio)
    eval_size = max(min_eval, eval_size)
    eval_size = min(max_eval, eval_size)
    eval_size = min(eval_size, max(1, len(shuffled) // 10))
    eval_rows = shuffled[:eval_size]
    train_rows = shuffled[eval_size:]
    return train_rows, eval_rows


def main():
    gmod_source_datasets = load_gmod_source_datasets()
    lua_source_path, _ = load_lua_source_dataset()
    has_gmod_source = bool(gmod_source_datasets)
    has_lua_source = lua_source_path is not None
    gmod_rows = build_gmod_rows()
    lua_rows = build_lua_manual_rows()
    uploaded_rows = load_uploaded_training_rows()
    gmod_training_rows = deduplicate_rows(gmod_rows + uploaded_rows)
    combined_rows = gmod_training_rows + lua_rows
    train_rows, eval_rows = split_rows(combined_rows)
    focused_rows, quickstart_rows = build_focused_datasets(gmod_training_rows, lua_rows)
    coding_rows = build_coding_dataset(gmod_training_rows, lua_rows)
    focused_train_rows, focused_eval_rows = split_rows(focused_rows, eval_ratio=0.03, min_eval=100, max_eval=250)
    quickstart_train_rows, quickstart_eval_rows = split_rows(quickstart_rows, eval_ratio=0.04, min_eval=80, max_eval=160)
    coding_train_rows, coding_eval_rows = split_rows(coding_rows, eval_ratio=0.03, min_eval=80, max_eval=180)

    columns = ["question", "answer", "prompt", "completion", "source", "dataset", "title", "kind", "focus_tags", "score", "page_score"]
    outputs = {
        DATASETS_DIR / "gmod_reference.jsonl": gmod_rows,
        DATASETS_DIR / "gmod_reference.csv": gmod_rows,
        DATASETS_DIR / "lua51_reference.jsonl": lua_rows,
        DATASETS_DIR / "lua51_reference.csv": lua_rows,
        DATASETS_DIR / "gmod_lua_train.jsonl": train_rows,
        DATASETS_DIR / "gmod_lua_train.csv": train_rows,
        DATASETS_DIR / "gmod_lua_eval.jsonl": eval_rows,
        DATASETS_DIR / "gmod_lua_eval.csv": eval_rows,
        DATASETS_DIR / "gmod_lua_focused_train.jsonl": focused_train_rows,
        DATASETS_DIR / "gmod_lua_focused_train.csv": focused_train_rows,
        DATASETS_DIR / "gmod_lua_focused_eval.jsonl": focused_eval_rows,
        DATASETS_DIR / "gmod_lua_focused_eval.csv": focused_eval_rows,
        DATASETS_DIR / "gmod_lua_quickstart_train.jsonl": quickstart_train_rows,
        DATASETS_DIR / "gmod_lua_quickstart_train.csv": quickstart_train_rows,
        DATASETS_DIR / "gmod_lua_quickstart_eval.jsonl": quickstart_eval_rows,
        DATASETS_DIR / "gmod_lua_quickstart_eval.csv": quickstart_eval_rows,
        DATASETS_DIR / "gmod_lua_coding_train.jsonl": coding_train_rows,
        DATASETS_DIR / "gmod_lua_coding_train.csv": coding_train_rows,
        DATASETS_DIR / "gmod_lua_coding_eval.jsonl": coding_eval_rows,
        DATASETS_DIR / "gmod_lua_coding_eval.csv": coding_eval_rows,
    }

    for path, rows in outputs.items():
        if path.suffix == ".csv":
            write_csv(path, rows, columns)
        else:
            write_jsonl(path, rows)

    write_messages_jsonl(DATASETS_DIR / "gmod_lua_train_messages.jsonl", train_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_eval_messages.jsonl", eval_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_focused_train_messages.jsonl", focused_train_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_focused_eval_messages.jsonl", focused_eval_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_quickstart_train_messages.jsonl", quickstart_train_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_quickstart_eval_messages.jsonl", quickstart_eval_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_coding_train_messages.jsonl", coding_train_rows)
    write_messages_jsonl(DATASETS_DIR / "gmod_lua_coding_eval_messages.jsonl", coding_eval_rows)
    remove_legacy_unsloth_outputs()

    print(f"Built {len(gmod_rows)} GMod rows")
    print(f"Built {len(lua_rows)} Lua 5.1 rows")
    if uploaded_rows:
        print(f"Included {len(uploaded_rows)} uploaded file rows from {UPLOADED_FILES_TRAIN_PATH.relative_to(REPO_ROOT)}")
    if not has_gmod_source:
        print("No scraped GMod source files were found in datasets/scraped. Built a starter dataset from the curated in-repo rows instead.")
    if not has_lua_source:
        print("No Lua 5.1 scraped source file was found in datasets/scraped. Skipped Lua reference rows.")
    print(f"Wrote {len(train_rows)} training rows and {len(eval_rows)} evaluation rows")
    print(f"Focused dataset: {len(focused_train_rows)} train / {len(focused_eval_rows)} eval")
    print(f"Quickstart dataset: {len(quickstart_train_rows)} train / {len(quickstart_eval_rows)} eval")
    print(f"Coding dataset: {len(coding_train_rows)} train / {len(coding_eval_rows)} eval")


if __name__ == "__main__":
    main()