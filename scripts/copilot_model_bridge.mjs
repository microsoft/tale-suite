#!/usr/bin/env node

import { existsSync } from "node:fs";
import { homedir } from "node:os";
import { pathToFileURL } from "node:url";
import readline from "node:readline";

const sdkPath = process.env.COPILOT_SDK_PATH;
if (!sdkPath || !existsSync(sdkPath)) {
  throw new Error(
    "COPILOT_SDK_PATH must point to the Copilot SDK index.js file.",
  );
}

const cliPath = process.env.COPILOT_CLI_PATH || "/usr/local/bin/copilot";
if (!existsSync(cliPath)) {
  throw new Error(`Copilot CLI runtime not found at ${cliPath}.`);
}

const { CopilotClient, RuntimeConnection, defineTool } = await import(
  pathToFileURL(sdkPath).href
);

const client = new CopilotClient({
  mode: "empty",
  baseDirectory: process.env.COPILOT_HOME || `${homedir()}/.copilot`,
  workingDirectory: process.env.TALES_REPO_ROOT || process.cwd(),
  connection: RuntimeConnection.forStdio({ path: cliPath }),
  logLevel: process.env.COPILOT_BRIDGE_LOG_LEVEL || "error",
});

const sessions = new Map();
let shuttingDown = false;

function writeMessage(message) {
  process.stdout.write(`${JSON.stringify(message)}\n`);
}

function errorPayload(error) {
  return {
    name: error?.name || "Error",
    message: error?.message || String(error),
    stack: error?.stack,
  };
}

function sanitizeUsage(data) {
  return {
    api_call_id: data.apiCallId,
    model: data.model,
    input_tokens: data.inputTokens || 0,
    output_tokens: data.outputTokens || 0,
    reasoning_tokens: data.reasoningTokens || 0,
    cache_read_tokens: data.cacheReadTokens || 0,
    cache_write_tokens: data.cacheWriteTokens || 0,
    duration_ms: data.duration || 0,
    reasoning_effort: data.reasoningEffort,
    finish_reason: data.finishReason,
    content_filter_triggered: Boolean(data.contentFilterTriggered),
    cost: data.cost || 0,
    aiu_nano: data.copilotUsage?.totalNanoAiu || 0,
  };
}

function sanitizeModel(model) {
  return {
    id: model.id,
    name: model.name,
    family: model.capabilities?.family || model.id,
    policy_state: model.policy?.state,
    supported_reasoning_efforts: model.supportedReasoningEfforts || [],
    default_reasoning_effort: model.defaultReasoningEffort,
    context_window: model.capabilities?.limits?.max_context_window_tokens,
    max_prompt_tokens: model.capabilities?.limits?.max_prompt_tokens,
    max_output_tokens: model.capabilities?.limits?.max_output_tokens,
    billing: model.billing || {},
    model_picker_category: model.modelPickerCategory,
    model_picker_price_category: model.modelPickerPriceCategory,
  };
}

function gameActionTool(state) {
  return defineTool("game_action", {
    description:
      "Submit exactly one short command to the current text adventure game.",
    parameters: {
      type: "object",
      properties: {
        action: {
          type: "string",
          minLength: 1,
          maxLength: 198,
          description:
            "One executable game command, such as 'take lamp', 'inventory', or 'north'. For a keypress prompt, use one literal space for SPACE/any key or '<ENTER>' for Enter/Return. Never use backslash escape sequences or key codes.",
        },
      },
      required: ["action"],
      additionalProperties: false,
    },
    skipPermission: true,
    defer: "never",
    handler: async (args, invocation) => {
      state.toolCalls.push({
        action: args?.action,
        tool_call_id: invocation.toolCallId,
      });
      state.session?.abort().catch(() => {});
      return {
        textResultForLlm: "Action recorded. Wait for the next game observation.",
        resultType: "success",
      };
    },
  });
}

function normalizeGameAction(action) {
  if (typeof action !== "string") {
    return "";
  }
  if (action.length === 0) {
    return "<ENTER>";
  }
  if (/^ +$/.test(action) || /^<space>$/iu.test(action.trim())) {
    return " ";
  }
  if (/^<(enter|return)>$/iu.test(action.trim())) {
    return "<ENTER>";
  }
  return action.trim();
}

function serializeToolCalls(toolCalls, assistantMessages) {
  if (toolCalls.length === 0) {
    return "";
  }
  const toolRequests = assistantMessages.flatMap(
    (event) => event.data.tool_requests,
  );
  const requests = toolCalls.map(
    (toolCall) =>
      toolRequests.find(
        (requestEvent) => requestEvent?.toolCallId === toolCall.tool_call_id,
      ) || {
        toolCallId: toolCall.tool_call_id,
        name: "game_action",
        arguments: { action: toolCall.action },
        type: "function",
      },
  );
  return JSON.stringify(requests.length === 1 ? requests[0] : requests);
}

function sessionConfig(request, includeSessionId, state) {
  const textMode = request.response_mode === "text";
  const config = {
    clientName: textMode
      ? "tale-suite-jaz-jericho"
      : "tale-suite-jericho-evals",
    model: request.model,
    allowedModels: [request.model],
    reasoningSummary: request.reasoning_summary || "none",
    tools: textMode ? [] : [gameActionTool(state)],
    availableTools: textMode ? [] : ["game_action"],
    toolSearch: { enabled: false },
    systemMessage: {
      mode: "replace",
      content: request.system_prompt,
    },
    enableConfigDiscovery: false,
    skipCustomInstructions: true,
    requestCanvasRenderer: false,
    requestExtensions: false,
    coauthorEnabled: false,
    manageScheduleEnabled: false,
  };

  if (textMode) {
    config.infiniteSessions = { enabled: false };
    config.memory = { enabled: false };
  }

  if (request.reasoning_effort) {
    config.reasoningEffort = request.reasoning_effort;
  }
  if (includeSessionId && request.session_id) {
    config.sessionId = request.session_id;
  }
  return config;
}

function registerSession(state, session) {
  state.session = session;
  session.on((event) => {
    if (event.agentId) {
      return;
    }
    if (event.type === "assistant.usage") {
      state.events.push({
        type: event.type,
        data: sanitizeUsage(event.data),
      });
    } else if (event.type === "assistant.reasoning") {
      state.events.push({
        type: event.type,
        data: { content: event.data.content || "" },
      });
    } else if (event.type === "assistant.message") {
      state.events.push({
        type: event.type,
        data: {
          content: event.data.content || "",
          reasoning_text: event.data.reasoningText || "",
          model: event.data.model,
          tool_requests: event.data.toolRequests || [],
        },
      });
    } else if (event.type === "assistant.turn_end") {
      state.events.push({
        type: event.type,
        data: { turn_id: event.data.turnId },
      });
      if (state.responseMode !== "text" && state.toolCalls.length === 0) {
        state.session?.abort().catch(() => {});
      }
    } else if (event.type === "session.error") {
      state.events.push({
        type: event.type,
        data: event.data,
      });
    }
  });
  sessions.set(session.sessionId, state);
  return state;
}

async function enforceRequestedModel(session, model) {
  const policy = await session.rpc.model.setAllowedModels({
    allowedModels: [model],
  });
  for (const [field, value] of [
    ["modelId", policy.modelId],
    ["fallbackModel", policy.fallbackModel],
  ]) {
    if (value && value !== model) {
      throw new Error(
        `Exact-model policy mismatch: requested ${model}, ${field}=${value}.`,
      );
    }
  }
  return policy;
}

async function createSession(request) {
  const state = {
    session: null,
    events: [],
    toolCalls: [],
    responseMode: request.response_mode === "text" ? "text" : "tool",
  };
  const session = await withReasoningSummaryFallback(request, (config) =>
    client.createSession(config),
  state, true);
  const modelPolicy = await enforceRequestedModel(session, request.model);
  registerSession(state, session);
  return {
    session_id: session.sessionId,
    resumed: false,
    model_policy: modelPolicy,
  };
}

async function resumeSession(request) {
  const state = {
    session: null,
    events: [],
    toolCalls: [],
    responseMode: request.response_mode === "text" ? "text" : "tool",
  };
  const session = await withReasoningSummaryFallback(
    request,
    (config) => client.resumeSession(request.session_id, config),
    state,
    false,
  );
  const modelPolicy = await enforceRequestedModel(session, request.model);
  registerSession(state, session);
  return {
    session_id: session.sessionId,
    resumed: true,
    model_policy: modelPolicy,
  };
}

async function withReasoningSummaryFallback(request, open, state, includeSessionId) {
  const config = sessionConfig(request, includeSessionId, state);
  try {
    return await open(config);
  } catch (error) {
    if (!config.reasoningSummary || config.reasoningSummary === "none") {
      throw error;
    }
    const message = String(error?.message || error).toLowerCase();
    if (
      !message.includes("reasoning") ||
      (!message.includes("summary") && !message.includes("unsupported"))
    ) {
      throw error;
    }
    // Some models reject configurable reasoning summaries; retry without one.
    delete config.reasoningSummary;
    return open(config);
  }
}

async function closeSession(sessionId, shouldDelete) {
  const state = sessions.get(sessionId);
  if (state) {
    await state.session.disconnect();
    sessions.delete(sessionId);
  }
  if (shouldDelete) {
    await client.deleteSession(sessionId);
  }
  return { session_id: sessionId, deleted: Boolean(shouldDelete) };
}

function findIncompleteUserTurn(events) {
  let userIndex = -1;
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index];
    if (!event.agentId && event.type === "user.message") {
      userIndex = index;
      break;
    }
  }
  if (userIndex < 0) {
    return null;
  }

  const suffix = events.slice(userIndex + 1).filter((event) => !event.agentId);
  const completed = suffix.some(
    (event) =>
      event.type === "assistant.turn_end" ||
      (event.type === "assistant.message" &&
        (event.data?.toolRequests || []).some(
          (request) => request?.name === "game_action",
        )),
  );
  return completed ? null : events[userIndex];
}

async function repairIncompleteTurn(sessionId) {
  const state = sessions.get(sessionId);
  if (!state) {
    throw new Error(`No open session ${sessionId}.`);
  }
  const events = await state.session.getEvents();
  const boundary = findIncompleteUserTurn(events);
  if (!boundary) {
    return { repaired: false, events_removed: 0 };
  }
  const result = await state.session.rpc.history.truncate({
    eventId: boundary.id,
  });
  return {
    repaired: true,
    event_id: boundary.id,
    events_removed: result.eventsRemoved || 0,
    checkpoint_cleanup_failed: Boolean(result.checkpointCleanupFailed),
  };
}

async function sendMessage(request) {
  const state = sessions.get(request.session_id);
  if (!state) {
    throw new Error(`No open session ${request.session_id}.`);
  }

  // Reset the per-turn buffers so a 300 step episode cannot grow unbounded.
  state.events.length = 0;
  state.toolCalls.length = 0;
  const response = await state.session.sendAndWait(
    { prompt: request.prompt },
    request.timeout_ms,
  );

  const newEvents = state.events.slice();
  const toolCalls = state.toolCalls.slice();

  const usageEvents = newEvents
    .filter((event) => event.type === "assistant.usage")
    .map((event) => event.data);
  const usageByCall = new Map();
  for (const [index, item] of usageEvents.entries()) {
    usageByCall.set(item.api_call_id || `usage-event-${index}`, item);
  }
  const usage = [...usageByCall.values()];
  const reasoningEventParts = newEvents
    .filter((event) => event.type === "assistant.reasoning")
    .map((event) => event.data.content)
    .filter(Boolean);
  const assistantMessages = newEvents.filter(
    (event) => event.type === "assistant.message",
  );
  const messageReasoningParts = assistantMessages
    .map((message) => message.data.reasoning_text)
    .filter(Boolean);
  const reasoningParts =
    reasoningEventParts.length > 0
      ? reasoningEventParts
      : messageReasoningParts;
  const sessionErrors = newEvents.filter(
    (event) => event.type === "session.error",
  );
  if (sessionErrors.length) {
    throw new Error(
      `Copilot session error: ${JSON.stringify(sessionErrors.at(-1).data)}`,
    );
  }

  const assistantTextRaw = assistantMessages
    .map((event) => event.data.content)
    .filter(Boolean)
    .join("\n");
  const assistantText = assistantTextRaw.trim();
  const thinking = reasoningParts.join("\n").trim() || null;
  const reportedModel =
    response?.data?.model ||
    assistantMessages.at(-1)?.data?.model ||
    null;
  const model = reportedModel || request.model;

  if (state.responseMode === "text") {
    return {
      status: "ok",
      text: assistantTextRaw,
      thinking,
      assistant_text: assistantTextRaw,
      model,
      reported_model: reportedModel,
      usage,
    };
  }

  // A turn without a tool call still burned an API call: report it (with its
  // usage) instead of throwing, so the caller can nudge and keep telemetry.
  const action = toolCalls.length === 1 ? toolCalls[0].action : null;
  const normalizedAction = normalizeGameAction(action);
  const rawOutput = serializeToolCalls(toolCalls, assistantMessages);
  let invalidStatus = "invalid_tool_action";
  if (usage.length > 1) {
    invalidStatus = "multiple_api_calls";
  } else if (usage.length === 0) {
    invalidStatus = "missing_usage";
  } else if (toolCalls.length > 1) {
    invalidStatus = "multiple_tool_calls";
  } else if (toolCalls.length === 0) {
    invalidStatus = "no_tool_call";
  }
  if (
    usage.length !== 1 ||
    !normalizedAction ||
    Buffer.byteLength(normalizedAction, "utf8") > 198 ||
    normalizedAction.startsWith("\\") ||
    /[\p{Cc}\p{Cf}\p{Cs}\p{Zl}\p{Zp}]/u.test(action)
  ) {
    return {
      status: invalidStatus,
      action: typeof action === "string" ? action : "",
      raw_output: rawOutput,
      thinking,
      assistant_text: assistantText,
      model,
      usage,
      extra_tool_calls: Math.max(toolCalls.length - 1, 0),
    };
  }

  return {
    status: "ok",
    action: normalizedAction,
    raw_output: rawOutput,
    thinking,
    assistant_text: assistantText,
    model,
    usage,
    extra_tool_calls: Math.max(toolCalls.length - 1, 0),
  };
}

async function shutdown() {
  if (shuttingDown) {
    return { stopped: true };
  }
  shuttingDown = true;
  for (const state of sessions.values()) {
    try {
      await state.session.disconnect();
    } catch (error) {
      console.error(error);
    }
  }
  sessions.clear();
  const errors = await client.stop();
  return {
    stopped: true,
    errors: errors.map((error) => errorPayload(error)),
  };
}

async function handle(request) {
  switch (request.command) {
    case "ping":
      return { pong: true };
    case "list_models":
      return {
        models: (await client.listModels()).map((model) =>
          sanitizeModel(model),
        ),
      };
    case "create_session":
      return createSession(request);
    case "resume_session":
      return resumeSession(request);
    case "repair_incomplete_turn":
      return repairIncompleteTurn(request.session_id);
    case "send":
      return sendMessage(request);
    case "close_session":
      return closeSession(request.session_id, request.delete !== false);
    case "shutdown":
      return shutdown();
    default:
      throw new Error(`Unknown bridge command: ${request.command}`);
  }
}

await client.start();
writeMessage({ type: "ready" });

const input = readline.createInterface({
  input: process.stdin,
  crlfDelay: Infinity,
});

const inFlight = new Set();

input.on("line", (line) => {
  if (!line.trim() || shuttingDown) {
    return;
  }

  let request;
  try {
    request = JSON.parse(line);
  } catch (error) {
    writeMessage({ id: null, ok: false, error: errorPayload(error) });
    return;
  }

  const pending = Promise.resolve(handle(request))
    .then((result) => {
      writeMessage({ id: request.id, ok: true, result });
      if (request.command === "shutdown") {
        input.close();
      }
    })
    .catch((error) => {
      writeMessage({
        id: request.id,
        ok: false,
        error: errorPayload(error),
      });
    })
    .finally(() => {
      inFlight.delete(pending);
    });
  inFlight.add(pending);
});

input.on("close", async () => {
  // Drain in-flight requests so a closed stdin never drops a pending answer.
  await Promise.allSettled([...inFlight]);
  if (!shuttingDown) {
    await shutdown();
  }
  process.exit(0);
});

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.on(signal, async () => {
    await shutdown();
    process.exit(0);
  });
}
