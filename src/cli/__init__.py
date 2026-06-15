"""SearchClaw CLI — a terminal front-end for the web-research agent.

The CLI is a second consumer of the same `query_loop` agent core used by
the web server. It calls the loop in-process and renders the StreamEvent
flow to the terminal instead of a WebSocket.
"""
