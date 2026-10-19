"use client";

import { useCallback, useEffect, useReducer, useRef } from "react";

import { ApiError, UnreachableError, classifyStream } from "@/lib/api";
import { STAGES } from "@/lib/stages";
import type { ClassifyResponse, Stage, StageEvent, StreamEvent } from "@/lib/types";

export interface StageState {
  status: "pending" | StageEvent["status"];
  detail: string | null;
  elapsedMs: number | null;
}

export interface ClassifierError {
  message: string;
  unreachable: boolean;
}

export interface ClassifierState {
  phase: "idle" | "running" | "done" | "error" | "cancelled";
  stages: Record<Stage, StageState>;
  result: ClassifyResponse | null;
  error: ClassifierError | null;
}

type Action =
  | { type: "start" }
  | { type: "event"; event: StreamEvent }
  | { type: "streamEnd" }
  | { type: "fail"; error: ClassifierError }
  | { type: "cancel" };

const pendingStages = (): Record<Stage, StageState> =>
  Object.fromEntries(
    STAGES.map(({ id }) => [id, { status: "pending", detail: null, elapsedMs: null }]),
  ) as Record<Stage, StageState>;

const initialState: ClassifierState = { phase: "idle", stages: pendingStages(), result: null, error: null };

/** Stops the spinner on stages that were still running when the run ended early. */
function settle(stages: ClassifierState["stages"], status: "pending" | "error", detail: string) {
  return Object.fromEntries(
    Object.entries(stages).map(([id, s]) => [id, s.status === "running" ? { ...s, status, detail } : s]),
  ) as ClassifierState["stages"];
}

function reducer(state: ClassifierState, action: Action): ClassifierState {
  switch (action.type) {
    case "start":
      return { ...initialState, phase: "running", stages: pendingStages() };
    case "cancel":
      return { ...state, phase: "cancelled", stages: settle(state.stages, "pending", "Cancelled") };
    case "fail":
      return { ...state, phase: "error", stages: settle(state.stages, "error", "Interrupted"), error: action.error };
    case "streamEnd":
      // A well-behaved stream always ends with a result or error event.
      return state.phase === "running"
        ? {
            ...state,
            phase: "error",
            stages: settle(state.stages, "error", "Interrupted"),
            error: { message: "The stream ended before a result arrived.", unreachable: false },
          }
        : state;
    case "event": {
      const { event } = action;
      if (event.event === "stage") {
        const stage: StageState = { status: event.status, detail: event.detail, elapsedMs: event.elapsed_ms };
        return { ...state, stages: { ...state.stages, [event.stage]: stage } };
      }
      if (event.event === "error") {
        return {
          ...state,
          phase: "error",
          stages: settle(state.stages, "error", "Failed"),
          error: { message: event.message, unreachable: false },
        };
      }
      // The final response carries authoritative timings; fill in anything the stream did not show.
      const stages = { ...state.stages };
      for (const t of event.data.stages) {
        stages[t.stage] = { status: t.status, detail: t.detail, elapsedMs: t.elapsed_ms };
      }
      return { ...state, phase: "done", stages, result: event.data };
    }
  }
}

function toClassifierError(err: unknown): ClassifierError {
  if (err instanceof UnreachableError) return { message: err.message, unreachable: true };
  if (err instanceof ApiError) return { message: err.message, unreachable: false };
  return { message: "The connection to the backend was lost before the result arrived.", unreachable: true };
}

export function useClassifier() {
  const [state, dispatch] = useReducer(reducer, initialState);
  const abortRef = useRef<AbortController | null>(null);

  const cancel = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  const run = useCallback(async (form: FormData) => {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    const { signal } = controller;

    dispatch({ type: "start" });
    try {
      await classifyStream(form, (event) => !signal.aborted && dispatch({ type: "event", event }), signal);
      if (!signal.aborted) dispatch({ type: "streamEnd" });
    } catch (err) {
      // A newer run replaced this one: it owns the state now.
      if (abortRef.current !== controller) return;
      dispatch(signal.aborted ? { type: "cancel" } : { type: "fail", error: toClassifierError(err) });
    }
  }, []);

  useEffect(() => () => abortRef.current?.abort(), []);

  return { state, run, cancel };
}
