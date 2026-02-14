import React, { useEffect, useMemo, useState } from "react";
import {
  Button,
  Modal,
  Pressable,
  ScrollView,
  Text,
  TextInput,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import {
  createEvent,
  createRecurringExposure,
  fetchRecurringExposures,
  ingestTextEvent,
} from "../api/client";
import { RecurringExposureRule } from "../models/events";
import { useAuth } from "../auth/AuthContext";

const FONT_SEMIBOLD = "Exo2-SemiBold";
const FONT_BOLD = "Exo2-Bold";
const ROUTE_OPTIONS = [
  { value: "ingestion", label: "Ingestion" },
  { value: "dermal", label: "Topical / Dermal" },
  { value: "inhalation", label: "Inhalation" },
  { value: "injection", label: "Injection" },
  { value: "proximity_environment", label: "Proximity / Environment" },
  { value: "other", label: "Other" },
] as const;

function formatIsoForDisplay(iso: string): string {
  const dt = new Date(iso);
  if (Number.isNaN(dt.getTime())) return iso;
  return dt.toLocaleString();
}

function toIsoFromParts(date: Date): string {
  return new Date(date.getTime() - date.getTimezoneOffset() * 60000).toISOString();
}

function clampDay(year: number, month1: number, day: number): number {
  const maxDay = new Date(year, month1, 0).getDate();
  return Math.max(1, Math.min(maxDay, day));
}

function shiftDatePart(date: Date, part: "year" | "month" | "day" | "hour" | "minute", delta: number): Date {
  const d = new Date(date);
  if (part === "year") {
    const year = d.getFullYear() + delta;
    const month = d.getMonth() + 1;
    const day = clampDay(year, month, d.getDate());
    d.setFullYear(year, month - 1, day);
    return d;
  }
  if (part === "month") {
    const next = new Date(d.getFullYear(), d.getMonth() + delta, 1, d.getHours(), d.getMinutes(), 0, 0);
    const day = clampDay(next.getFullYear(), next.getMonth() + 1, d.getDate());
    next.setDate(day);
    return next;
  }
  if (part === "day") {
    d.setDate(d.getDate() + delta);
    return d;
  }
  if (part === "hour") {
    d.setHours(d.getHours() + delta);
    return d;
  }
  d.setMinutes(d.getMinutes() + delta);
  return d;
}

export default function LogEventScreen() {
  const { user } = useAuth();
  const [itemName, setItemName] = useState("");
  const [route, setRoute] = useState<(typeof ROUTE_OPTIONS)[number]["value"]>("ingestion");
  const [repeatDays, setRepeatDays] = useState<number>(0); // 0 = one-time
  const [startDate, setStartDate] = useState<Date>(() => new Date());
  const [pickerOpen, setPickerOpen] = useState(false);
  const [routePickerOpen, setRoutePickerOpen] = useState(false);
  const [repeatPickerOpen, setRepeatPickerOpen] = useState(false);
  const [rawText, setRawText] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  const [textStatus, setTextStatus] = useState<string | null>(null);
  const [, setRecurringRules] = useState<RecurringExposureRule[]>([]);

  const startIso = useMemo(() => toIsoFromParts(startDate), [startDate]);
  const repeatOptions = [0, 1, 2, 3, 4, 5, 6, 7];

  const selectedRouteLabel = useMemo(
    () => ROUTE_OPTIONS.find((option) => option.value === route)?.label ?? "Other",
    [route]
  );
  const selectedRepeatLabel = useMemo(
    () => (repeatDays === 0 ? "One time" : `${repeatDays} day${repeatDays === 1 ? "" : "s"}`),
    [repeatDays]
  );

  const loadRecurring = React.useCallback(async () => {
    if (!user) return;
    const rows = await fetchRecurringExposures(user.id);
    setRecurringRules(rows);
  }, [user]);

  useEffect(() => {
    loadRecurring().catch(() => undefined);
  }, [loadRecurring]);

  useEffect(() => {
    if (!error) return;
    const timer = setTimeout(() => setError(null), 5000);
    return () => clearTimeout(timer);
  }, [error]);

  async function submitExposurePlan() {
    setError(null);
    setStatus(null);
    setTextStatus(null);
    if (!itemName.trim()) {
      setError("Exposure item is required.");
      return;
    }
    try {
      if (!user) {
        setError("Not authenticated.");
        return;
      }
      if (repeatDays <= 0) {
        const res = await createEvent({
          event_type: "exposure",
          user_id: user.id,
          timestamp: startIso,
          item_name: itemName.trim(),
          route,
          time_confidence: "exact",
        });
        if (res.status === "queued") {
          setStatus(`Queued one-time event: ${res.resolution ?? "pending"}`);
        } else {
          setStatus("Logged one-time exposure.");
        }
      } else {
        await createRecurringExposure({
          user_id: user.id,
          item_name: itemName.trim(),
          route,
          start_at: startIso,
          interval_hours: repeatDays * 24,
          time_confidence: "approx",
        });
        setStatus(`Saved recurring exposure every ${repeatDays} day(s).`);
        await loadRecurring();
      }
      setItemName("");
      setRepeatDays(0);
    } catch (err: any) {
      setError(err?.message ?? "Failed to save exposure plan.");
    }
  }

  async function submitText() {
    setError(null);
    setStatus(null);
    setTextStatus(null);
    if (!rawText.trim()) {
      setError("Text input is required.");
      return;
    }
    try {
      if (!user) {
        setError("Not authenticated.");
        return;
      }
      await ingestTextEvent({ user_id: user.id, raw_text: rawText });
      setTextStatus("Logged!");
      setRawText("");
    } catch (err: any) {
      setError(err?.message ?? "Failed to ingest text.");
    }
  }

  return (
    <SafeAreaView style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={{ padding: 16, gap: 14 }}>
        <Text style={{ fontSize: 18, fontFamily: FONT_SEMIBOLD }}>Log Event</Text>

        <View style={{ padding: 12, borderWidth: 1, borderColor: "#E1E5EE", borderRadius: 10, gap: 10 }}>
          <Text style={{ fontSize: 16, fontFamily: FONT_SEMIBOLD }}>Log an Exposure</Text>
          <TextInput
            style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
            placeholder="Exposure item (e.g. coffee)"
            value={itemName}
            onChangeText={setItemName}
          />

          <Pressable
            onPress={() => setRoutePickerOpen(true)}
            style={{
              borderWidth: 1,
              borderColor: "#C9CFDE",
              borderRadius: 8,
              padding: 10,
              backgroundColor: "#FAFBFF",
            }}
          >
            <Text style={{ color: "#39445F", fontFamily: FONT_SEMIBOLD }}>Choose route</Text>
            <Text style={{ color: "#5C6784", marginTop: 2 }}>{selectedRouteLabel}</Text>
          </Pressable>

          <Pressable
            onPress={() => setPickerOpen(true)}
            style={{
              borderWidth: 1,
              borderColor: "#C9CFDE",
              borderRadius: 8,
              padding: 10,
              backgroundColor: "#FAFBFF",
            }}
          >
            <Text style={{ color: "#39445F", fontFamily: FONT_SEMIBOLD }}>Start Date + Time</Text>
            <Text style={{ color: "#5C6784", marginTop: 2 }}>{formatIsoForDisplay(startIso)}</Text>
          </Pressable>

          <Pressable
            onPress={() => setRepeatPickerOpen(true)}
            style={{
              borderWidth: 1,
              borderColor: "#C9CFDE",
              borderRadius: 8,
              padding: 10,
              backgroundColor: "#FAFBFF",
            }}
          >
            <Text style={{ color: "#39445F", fontFamily: FONT_SEMIBOLD }}>Repeat every</Text>
            <Text style={{ color: "#5C6784", marginTop: 2 }}>{selectedRepeatLabel}</Text>
          </Pressable>

          <Button
            title={repeatDays === 0 ? "Log One-Time Exposure" : "Save Recurring Exposure"}
            onPress={submitExposurePlan}
          />
          {status ? <Text style={{ color: "#0A7A4F" }}>{status}</Text> : null}
        </View>

        <View style={{ padding: 12, borderWidth: 1, borderColor: "#E1E5EE", borderRadius: 10 }}>
          <Text style={{ fontSize: 16, fontFamily: FONT_SEMIBOLD, marginBottom: 6 }}>Quick Log (Text)</Text>
          <TextInput
            style={{
              borderWidth: 1,
              borderColor: "#ccc",
              padding: 10,
              borderRadius: 8,
              minHeight: 120,
              textAlignVertical: "top",
            }}
            value={rawText}
            onChangeText={setRawText}
            placeholder="What's new?"
            multiline
          />
          <View style={{ marginTop: 10 }}>
            <Button title="Log Text" onPress={submitText} />
          </View>
          {textStatus ? <Text style={{ marginTop: 8, color: "#0A7A4F" }}>{textStatus}</Text> : null}
        </View>

        {error ? <Text style={{ color: "#B42318" }}>{error}</Text> : null}
      </ScrollView>

      <Modal visible={pickerOpen} transparent animationType="fade" onRequestClose={() => setPickerOpen(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(16,24,40,0.35)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "white", borderRadius: 14, padding: 14, gap: 10 }}>
            <Text style={{ fontSize: 16, fontFamily: FONT_BOLD }}>Pick Start Date + Time</Text>
            {(
              [
                ["year", String(startDate.getFullYear())],
                ["month", String(startDate.getMonth() + 1).padStart(2, "0")],
                ["day", String(startDate.getDate()).padStart(2, "0")],
                ["hour", String(startDate.getHours()).padStart(2, "0")],
                ["minute", String(startDate.getMinutes()).padStart(2, "0")],
              ] as [("year" | "month" | "day" | "hour" | "minute"), string][]
            ).map(([part, value]) => (
              <View key={`picker-${part}`} style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                <Text style={{ textTransform: "capitalize", color: "#344054" }}>{part}</Text>
                <View style={{ flexDirection: "row", gap: 8, alignItems: "center" }}>
                  <Pressable
                    onPress={() => setStartDate((d) => shiftDatePart(d, part, -1))}
                    style={{ paddingHorizontal: 10, paddingVertical: 6, borderWidth: 1, borderColor: "#D0D5DD", borderRadius: 8 }}
                  >
                    <Text>-</Text>
                  </Pressable>
                  <Text style={{ minWidth: 56, textAlign: "center", fontFamily: FONT_SEMIBOLD }}>{value}</Text>
                  <Pressable
                    onPress={() => setStartDate((d) => shiftDatePart(d, part, 1))}
                    style={{ paddingHorizontal: 10, paddingVertical: 6, borderWidth: 1, borderColor: "#D0D5DD", borderRadius: 8 }}
                  >
                    <Text>+</Text>
                  </Pressable>
                </View>
              </View>
            ))}
            <Text style={{ color: "#475467", marginTop: 2 }}>{formatIsoForDisplay(startIso)}</Text>
            <View style={{ flexDirection: "row", gap: 10, marginTop: 6 }}>
              <View style={{ flex: 1 }}>
                <Button title="Done" onPress={() => setPickerOpen(false)} />
              </View>
              <View style={{ flex: 1 }}>
                <Button
                  title="Now"
                  onPress={() => {
                    setStartDate(new Date());
                    setPickerOpen(false);
                  }}
                />
              </View>
            </View>
          </View>
        </View>
      </Modal>
      <Modal visible={routePickerOpen} transparent animationType="fade" onRequestClose={() => setRoutePickerOpen(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(16,24,40,0.35)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "white", borderRadius: 14, padding: 14, gap: 10 }}>
            <Text style={{ fontSize: 16, fontFamily: FONT_BOLD }}>Choose Route</Text>
            <ScrollView style={{ maxHeight: 300 }}>
              {ROUTE_OPTIONS.map((option) => (
                <Pressable
                  key={`route-option-${option.value}`}
                  onPress={() => {
                    setRoute(option.value);
                    setRoutePickerOpen(false);
                  }}
                  style={{
                    borderWidth: 1,
                    borderColor: route === option.value ? "#2E5BCE" : "#D0D5DD",
                    borderRadius: 10,
                    paddingVertical: 10,
                    paddingHorizontal: 12,
                    backgroundColor: route === option.value ? "#EEF3FF" : "#FFF",
                    marginBottom: 8,
                  }}
                >
                  <Text style={{ color: route === option.value ? "#2E5BCE" : "#344054", fontFamily: FONT_SEMIBOLD }}>
                    {option.label}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>
            <View style={{ flexDirection: "row", gap: 10, marginTop: 4 }}>
              <View style={{ flex: 1 }}>
                <Button title="Done" onPress={() => setRoutePickerOpen(false)} />
              </View>
            </View>
          </View>
        </View>
      </Modal>
      <Modal visible={repeatPickerOpen} transparent animationType="fade" onRequestClose={() => setRepeatPickerOpen(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(16,24,40,0.35)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "white", borderRadius: 14, padding: 14, gap: 10 }}>
            <Text style={{ fontSize: 16, fontFamily: FONT_BOLD }}>Repeat every</Text>
            <ScrollView style={{ maxHeight: 300 }}>
              {repeatOptions.map((days) => {
                const label = days === 0 ? "One time" : `${days} day${days === 1 ? "" : "s"}`;
                const selected = repeatDays === days;
                return (
                  <Pressable
                    key={`repeat-option-${days}`}
                    onPress={() => {
                      setRepeatDays(days);
                      setRepeatPickerOpen(false);
                    }}
                    style={{
                      borderWidth: 1,
                      borderColor: selected ? "#2E5BCE" : "#D0D5DD",
                      borderRadius: 10,
                      paddingVertical: 10,
                      paddingHorizontal: 12,
                      backgroundColor: selected ? "#EEF3FF" : "#FFF",
                      marginBottom: 8,
                    }}
                  >
                    <Text style={{ color: selected ? "#2E5BCE" : "#344054", fontFamily: FONT_SEMIBOLD }}>
                      {label}
                    </Text>
                  </Pressable>
                );
              })}
            </ScrollView>
            <View style={{ flexDirection: "row", gap: 10, marginTop: 4 }}>
              <View style={{ flex: 1 }}>
                <Button title="Done" onPress={() => setRepeatPickerOpen(false)} />
              </View>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}
