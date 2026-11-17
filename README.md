# Alexein AI for Cursor

Gives Cursor your [Alexein](https://alexein.ai) memory: the agent searches what you have told Alexein before
answering, and saves new facts as you work, so the next session, and your other apps, pick up where this one left off.

The plugin adds one MCP server, `alexein-ai`, at `https://api.alexein.ai/v1/mcp`, and a rule, `alexein-ai`, that tells
the agent when to search and when to save. The full tool list is at
[docs.alexein.ai/mcp/tools](https://docs.alexein.ai/mcp/tools).

## Install

[Add to Cursor](cursor://anysphere.cursor-deeplink/mcp/install?name=alexein-ai&config=eyJ1cmwiOiJodHRwczovL2FwaS5hbGV4ZWluLmFpL3YxL21jcCJ9)

The link adds the MCP server. Once the plugin is listed in the Cursor Marketplace, you can also install it from there
by searching for **Alexein AI**, which adds the rule as well.

## Sign in

The first time Cursor connects to Alexein, it opens Alexein's sign-in in your browser. Sign in (or create a free
account at [memory.alexein.ai](https://memory.alexein.ai)), choose what the agent may do (read only, or read and
save), and select **Allow**. No API key is needed: the server uses OAuth 2.1 with dynamic client registration.

Everything saved can be seen, downloaded and erased in the Alexein app, under **My memory**. To disconnect, remove the
plugin or server, or disconnect Cursor in Alexein under **AI assistants**.

## Help

[docs.alexein.ai](https://docs.alexein.ai) · zainali@alexein.ai · [Privacy](https://alexein.ai/legal/privacy) ·
[Terms](https://alexein.ai/legal/terms)

## Recall and saving sessions (optional API key)

The plugin also adds hooks (`hooks/hooks.json`) and an always-on rule with the note on how to use the memory tools. Without an API key they only keep that rule; nothing is read or sent.

With a key in `ALEXEIN_API_KEY` (a personal key from **My memory › Keys** in the Alexein app, **Read and write**;
or `api_key` in `~/.config/alexein/hooks.json`):

- **Recall.** The first message of each session is looked up in your memory and the best matches are added as
  context, with the first tool the agent runs (Cursor's prompt hook can't add context). One search per session.
- **Saving sessions.** What you ask and the agent's final replies are saved to your memory, with the names of files
  changed and commands run (never their contents or output), five exchanges at a time. Anything that looks like a
  password, key or token is removed first. `ALEXEIN_CAPTURE=false` keeps recall and saves nothing.

`python3 <plugin>/scripts/alexein_hooks.py run cursor pause` (or `resume`, `status`) turns this off and on. The
hooks need `python3`; they are the same script as the Claude Code plugin's
([claude-plugins](https://github.com/Alexein-AI/claude-plugins)).
