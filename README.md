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
