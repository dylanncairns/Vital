import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { View, Text, FlatList, RefreshControl, TouchableOpacity, TextInput, Alert, Modal, Pressable, ScrollView, Button } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect } from "@react-navigation/native";
import { deleteEvent, fetchEventInsightLinks, fetchEvents, fetchInsights, setInsightRejection, setInsightVerification, updateEvent } from "../api/client";
import { EventInsightLink, Insight, TimelineEvent } from "../models/events";
import { styles } from "./TimelineScreen.styles";
import { useAuth } from "../auth/AuthContext";

const ROUTE_OPTIONS = [
  { value: "ingestion", label: "Food / Drink (Ingestion)" },
  { value: "oral", label: "Oral medication / supplement" },
  { value: "behavioral", label: "Lifestyle / Physiology" },
  { value: "dermal", label: "Topical / Dermal" },
  { value: "inhalation", label: "Inhalation" },
  { value: "intranasal", label: "Nasal / Sinus" },
  { value: "injection", label: "Injection" },
  { value: "proximity_environment", label: "Proximity / Environment" },
  { value: "other", label: "Other" },
] as const;
type TimelineRow =
  | { type: "date"; key: string; label: string }
  | { type: "event"; key: string; event: TimelineEvent };

// Display list on screen in evenets entity and show refresh spinner when refreshing
export default function TimelineScreen() {
    const { user } = useAuth();
    const [events, setEvents] = useState<TimelineEvent[]>([]);
    const [refreshing, setRefreshing] = useState(false);
    const [editingKey, setEditingKey] = useState<string | null>(null);
    const [editDate, setEditDate] = useState<Date>(() => new Date());
    const [editSeverity, setEditSeverity] = useState("");
    const [editRoute, setEditRoute] = useState("");
    const [editTimePickerOpen, setEditTimePickerOpen] = useState(false);
    const [editRoutePickerOpen, setEditRoutePickerOpen] = useState(false);
    const [busyKey, setBusyKey] = useState<string | null>(null);
    const [eventInsightMap, setEventInsightMap] = useState<Record<string, number[]>>({});
    const [insightsById, setInsightsById] = useState<Record<number, Insight>>({});
    const [selectedInsights, setSelectedInsights] = useState<Insight[]>([]);
    const [verifyingInsightId, setVerifyingInsightId] = useState<number | null>(null);
    const loadSeqRef = useRef(0);

    async function confirmAction(title: string, message: string): Promise<boolean> {
      if (typeof window !== "undefined" && typeof window.confirm === "function") {
        return window.confirm(`${title}\n\n${message}`);
      }
      return new Promise((resolve) => {
        Alert.alert(title, message, [
          { text: "Cancel", style: "cancel", onPress: () => resolve(false) },
          { text: "Delete", style: "destructive", onPress: () => resolve(true) },
        ]);
      });
    }

    function eventInsightKey(eventType: "exposure" | "symptom", eventId: number): string {
      return `${eventType}:${eventId}`;
    }

    function eventDate(value: string | null | undefined): Date | null {
      if (!value) return null;
      const parsed = new Date(value);
      return Number.isNaN(parsed.getTime()) ? null : parsed;
    }

    function formatIsoForDisplay(iso: string): string {
      const dt = new Date(iso);
      if (Number.isNaN(dt.getTime())) return iso;
      return dt.toLocaleString();
    }

    function toIsoFromParts(date: Date): string {
      // Keep the exact instant selected in local picker; backend stores UTC.
      return date.toISOString();
    }

    function upsertLocalEvent(
      eventType: "exposure" | "symptom",
      eventId: number,
      patch: Partial<TimelineEvent>
    ) {
      setEvents((prev) => prev.map((row) => (row.event_type === eventType && row.id === eventId ? { ...row, ...patch } : row)));
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

    function normalizeRouteValue(value: string | null | undefined): string {
      const raw = (value ?? "").trim().toLowerCase().replace(/\//g, "_").replace(/\s+/g, "_");
      if (!raw || raw === "unknown") return "other";
      if (raw === "oral" || raw === "sublingual" || raw === "buccal") return "ingestion";
      if (raw === "behavioral" || raw === "behavioural" || raw === "lifestyle" || raw === "physiology" || raw === "lifestyle_physiology") return "behavioral";
      if (raw === "topical") return "dermal";
      if (raw === "transdermal") return "dermal";
      if (raw === "intranasal" || raw === "nasal_spray") return "inhalation";
      if (raw === "intravenous" || raw === "intramuscular" || raw === "subcutaneous" || raw === "iv" || raw === "im" || raw === "subq" || raw === "sq") return "injection";
      if (raw === "proximity" || raw === "environment" || raw === "proximity_environment") return "proximity_environment";
      if (ROUTE_OPTIONS.some((option) => option.value === raw)) return raw;
      return "other";
    }

    const selectedEditRouteLabel = useMemo(
      () => ROUTE_OPTIONS.find((option) => option.value === normalizeRouteValue(editRoute))?.label ?? "Other",
      [editRoute]
    );
    const timelineHeaderTitle = useMemo(() => {
      const firstName = (user?.name ?? "").trim().split(/\s+/)[0] ?? "";
      if (!firstName) return "Your Timeline";
      const possessive = /s$/i.test(firstName) ? `${firstName}'` : `${firstName}'s`;
      return `${possessive} Timeline`;
    }, [user?.name]);

    function prettyLagBucket(bucket: string): string {
      const map: Record<string, string> = {
        "0_6h": "0-6 hours",
        "6_24h": "6-24 hours",
        "24_72h": "24-72 hours",
        "72h_7d": "72 hours-7 days",
      };
      return map[bucket] ?? bucket;
    }

    function formatEvidenceSummary(
      summary: string | null | undefined,
      opts?: { sourceLabel?: string; citationTitle?: string; abstractSnippet?: string | null }
    ): { summary: string; onset: string | null } {
      let raw = (summary ?? "No summary").trim();
      raw = raw
        .replace(/^\s*\d+\s+claim(?:s|\(s\))?\s+retrieved\s*[:;.\-]?\s*/i, "")
        .replace(/\b\d+\s+claim(?:s|\(s\))?\s+retrieved\s*[:;.\-]?\s*/gi, "")
        .replace(/\s{2,}/g, " ")
        .trim();
      const match = raw.match(/Dominant lag window:\s*([A-Za-z0-9_]+)\.?$/);
      const sourcePart = opts?.sourceLabel ? `from ${opts.sourceLabel} ` : "";
      const titlePart = opts?.citationTitle ? `(${opts.citationTitle}) ` : "";
      const snippetRaw = (opts?.abstractSnippet ?? "").replace(/\s+/g, " ").trim();
      const snippet = snippetRaw.length > 220 ? `${snippetRaw.slice(0, 217).trimEnd()}...` : snippetRaw;
      const snippetPart = snippet ? `${snippet} ` : "";
      const intro = snippet
        ? `Supportive evidence states that ${snippetPart}`
        : `Supportive evidence ${sourcePart}${titlePart}indicates that `;
      if (!match) {
        raw = raw.replace(/^overall evidence is supportive\s*(that)?\s*/i, intro);
        return { summary: raw, onset: null };
      }
      const bucket = match[1];
      const withoutLagSentence = raw.replace(/Dominant lag window:\s*[A-Za-z0-9_]+\.?$/, "").trim();
      const onsetText = `Most common onset in your data: ${prettyLagBucket(bucket)}`;
      const rewritten = withoutLagSentence.replace(
        /^overall evidence is supportive\s*(that)?\s*/i,
        intro
      );
      return { summary: rewritten, onset: onsetText };
    }

    function citationSourceLabel(citation: { source?: string | null; url?: string | null }): string {
      const source = (citation.source ?? "").trim();
      const sourceLower = source.toLowerCase();
      if (source && sourceLower !== "openai_file_search" && sourceLower !== "file_search") {
        return source;
      }
      const url = (citation.url ?? "").trim();
      if (!url) {
        return "Unknown Source";
      }
      try {
        return new URL(url).hostname || "Unknown Source";
      } catch {
        return "Unknown Source";
      }
    }

    const load = useCallback(async (showRefresh: boolean = false) => {
        if (!user) return;
        const seq = ++loadSeqRef.current;
        if (showRefresh) setRefreshing(true);
        try {
            const [data, links, insights] = await Promise.all([
              fetchEvents(user.id),
              fetchEventInsightLinks(user.id, true),
              fetchInsights(user.id, false),
            ]);
            const sorted = [...data].sort((a, b) => {
              const aDate = eventDate(a.timestamp);
              const bDate = eventDate(b.timestamp);
              if (!aDate && !bDate) return 0;
              if (!aDate) return 1;
              if (!bDate) return -1;

              // Date bucket sort: newer day first.
              const aDay = new Date(aDate.getFullYear(), aDate.getMonth(), aDate.getDate()).getTime();
              const bDay = new Date(bDate.getFullYear(), bDate.getMonth(), bDate.getDate()).getTime();
              if (aDay !== bDay) {
                return bDay - aDay;
              }

              // Within same day: morning -> night.
              const aSeconds = (aDate.getHours() * 3600) + (aDate.getMinutes() * 60) + aDate.getSeconds();
              const bSeconds = (bDate.getHours() * 3600) + (bDate.getMinutes() * 60) + bDate.getSeconds();
              return aSeconds - bSeconds;
            });
            if (seq !== loadSeqRef.current) return;
            setEvents(sorted);
            const insightLookup: Record<number, Insight> = {};
            for (const insight of insights) {
              insightLookup[insight.id] = insight;
            }
            setInsightsById(insightLookup);

            const mapped: Record<string, number[]> = {};
            for (const link of links as EventInsightLink[]) {
              if (!insightLookup[link.insight_id]) continue;
              const key = eventInsightKey(link.event_type, link.event_id);
              if (!mapped[key]) {
                mapped[key] = [];
              }
              mapped[key].push(link.insight_id);
            }
            for (const key of Object.keys(mapped)) {
              mapped[key].sort((a, b) => {
                const aScore = insightLookup[a]?.overall_confidence_score ?? 0;
                const bScore = insightLookup[b]?.overall_confidence_score ?? 0;
                return bScore - aScore;
              });
            }
            setEventInsightMap(mapped);
        } finally {
            if (showRefresh && seq === loadSeqRef.current) setRefreshing(false);
        }
    }, [user]);

    // Runs when screen first appears - triggers first API fetch to populate timeline
    useEffect(() => {
        load();
    }, [load]);

    useFocusEffect(
      useCallback(() => {
        load(false);
      }, [load])
    );

  const rows = useMemo<TimelineRow[]>(() => {
    const out: TimelineRow[] = [];
    let currentDateKey = "";
    for (const event of events) {
      const d = eventDate(event.timestamp);
      const dateKey = d
        ? `${d.getFullYear()}-${d.getMonth() + 1}-${d.getDate()}`
        : "unknown";
      if (dateKey !== currentDateKey) {
        currentDateKey = dateKey;
        out.push({
          type: "date",
          key: `date-${dateKey}`,
          label: d
            ? d.toLocaleDateString(undefined, {
                weekday: "short",
                month: "short",
                day: "numeric",
              })
            : "Unknown Date",
        });
      }
      out.push({
        type: "event",
        key: `event-${event.event_type}-${event.id}`,
        event,
      });
    }
    return out;
  }, [events]);

  return (
    <SafeAreaView style={styles.safe}>
      <FlatList
        data={rows}
        keyExtractor={(item) => item.key}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => load(true)} />}
        contentContainerStyle={[styles.container, { flexGrow: 1 }]}
        bounces={true}
        alwaysBounceVertical={true}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.headerTitle}>{timelineHeaderTitle}</Text>
          </View>
        }
        ListEmptyComponent={<Text style={styles.emptyText}>Add events to build your timeline.</Text>}
        renderItem={({ item }) => {
          if (item.type === "date") {
            return (
              <View style={styles.dateBreakRow}>
                <View style={styles.dateBreakLine} />
                <Text style={styles.dateBreakText}>{item.label}</Text>
              </View>
            );
          }

          const event = item.event;
          const isExposure = event.event_type === "exposure";
          const dotColor = isExposure ? "#00C389" : "#4F7BFF";
          const parsedTime = eventDate(event.timestamp);
          const time = parsedTime
            ? parsedTime.toLocaleTimeString([], {
                hour: "2-digit",
                minute: "2-digit",
              })
            : "Time unknown";
          const linkedInsightIds = eventInsightMap[eventInsightKey(event.event_type as "exposure" | "symptom", event.id)] ?? [];
          const linkedInsightCount = linkedInsightIds.length;
          return (
            <View style={styles.row}>
              <View style={styles.treeWrap}>
                <View style={styles.rail} />
                <View style={styles.nodeWrap}>
                  <View style={[styles.dot, { backgroundColor: dotColor }]} />
                  <View style={styles.branch} />
                </View>
              </View>

              <View style={styles.card}>
                <Text style={styles.timeText}>{time}</Text>
                <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                  <Text style={[styles.titleText, { flex: 1 }]}>
                    {isExposure
                      ? `Exposure: ${event.item_name ?? "Item"}`
                      : `Symptom: ${event.symptom_name ?? "Symptom"}`}
                  </Text>
                  {linkedInsightCount > 0 ? (
                    <TouchableOpacity
                      style={{
                        borderRadius: 999,
                        borderWidth: 1,
                        borderColor: "#2E5BCE",
                        backgroundColor: "#EEF3FF",
                        paddingHorizontal: 10,
                        paddingVertical: 5,
                      }}
                      onPress={() => {
                        const rows = linkedInsightIds
                          .map((id) => insightsById[id])
                          .filter((row): row is Insight => Boolean(row));
                        if (rows.length > 0) {
                          setSelectedInsights(rows);
                        }
                      }}
                    >
                      <Text style={{ color: "#2E5BCE", fontFamily: "Exo2-SemiBold", fontSize: 11 }}>
                        {linkedInsightCount === 1 ? "See 1 insight" : `See ${linkedInsightCount} insights`}
                      </Text>
                    </TouchableOpacity>
                  ) : null}
                </View>
                <Text style={styles.bodyText}>
                  {isExposure
                    ? `Route: ${(event.route ?? "other") === "unknown" ? "other" : (event.route ?? "other")}`
                    : `Severity: ${event.severity ?? "-"}`}
                </Text>
                {editingKey === item.key ? (
                  <View style={{ marginTop: 8, gap: 8 }}>
                    <Pressable
                      onPress={() => setEditTimePickerOpen(true)}
                      style={{
                        borderWidth: 1,
                        borderColor: "#C9CFDE",
                        borderRadius: 8,
                        padding: 10,
                        backgroundColor: "#FAFBFF",
                      }}
                    >
                      <Text style={{ color: "#343A52", fontFamily: "Exo2-SemiBold", fontSize: 12 }}>Event time</Text>
                      <Text style={{ color: "#343A52", marginTop: 2, fontFamily: "Exo2-Regular" }}>
                        {formatIsoForDisplay(toIsoFromParts(editDate))}
                      </Text>
                    </Pressable>
                    {isExposure ? (
                      <Pressable
                        onPress={() => setEditRoutePickerOpen(true)}
                        style={{
                          borderWidth: 1,
                          borderColor: "#C9CFDE",
                          borderRadius: 8,
                          padding: 10,
                          backgroundColor: "#FAFBFF",
                        }}
                      >
                        <Text style={{ color: "#343A52", fontFamily: "Exo2-SemiBold", fontSize: 12 }}>Route</Text>
                        <Text style={{ color: "#343A52", marginTop: 2, fontFamily: "Exo2-Regular" }}>
                          {selectedEditRouteLabel}
                        </Text>
                      </Pressable>
                    ) : (
                      <TextInput
                        style={{
                          borderWidth: 1,
                          borderColor: "#C9CFDE",
                          borderRadius: 8,
                          padding: 8,
                          color: "#343A52",
                          backgroundColor: "#FAFBFF",
                        }}
                        placeholderTextColor="#8E96B3"
                        placeholder="Severity (1-5)"
                        value={editSeverity}
                        onChangeText={setEditSeverity}
                      />
                    )}
                    {isExposure && ["unknown", "other"].includes(normalizeRouteValue(editRoute)) ? (
                      <Text style={{ color: "#5B6381", fontFamily: "Exo2-Regular", fontSize: 12 }}>
                        Adding route improves model accuracy and confidence.
                      </Text>
                    ) : null}
                    {!isExposure && !editSeverity.trim() ? (
                      <Text style={{ color: "#5B6381", fontFamily: "Exo2-Regular", fontSize: 12 }}>
                        Adding severity improves model accuracy and confidence.
                      </Text>
                    ) : null}
                    <View style={{ flexDirection: "row", gap: 10 }}>
                      <TouchableOpacity
                        style={{ backgroundColor: "#0A7A4F", borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8 }}
                        disabled={busyKey === item.key}
                        onPress={async () => {
                          setBusyKey(item.key);
                          try {
                            if (!user) {
                              Alert.alert("Not authenticated", "Please sign in again.");
                              return;
                            }
                            const nextTimestamp = toIsoFromParts(editDate);
                            const nextRoute = normalizeRouteValue(editRoute);
                            const nextSeverity = editSeverity.trim() ? Number(editSeverity) : undefined;
                            await updateEvent(
                              event.event_type,
                              event.id,
                              user.id,
                              isExposure
                                ? {
                                    timestamp: nextTimestamp,
                                    route: nextRoute,
                                  }
                                : {
                                    timestamp: nextTimestamp,
                                    severity: nextSeverity,
                                  }
                            );
                            upsertLocalEvent(event.event_type, event.id, {
                              timestamp: nextTimestamp,
                              ...(isExposure ? { route: nextRoute } : { severity: nextSeverity }),
                            });
                            setEditingKey(null);
                            setEditSeverity("");
                            setEditRoute("");
                            setEditTimePickerOpen(false);
                            setEditRoutePickerOpen(false);
                            void load(false);
                          } finally {
                            setBusyKey(null);
                          }
                        }}
                      >
                        <Text style={{ color: "white", fontFamily: "Exo2-SemiBold" }}>Save</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={{ backgroundColor: "#E9EDF6", borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8 }}
                        onPress={() => {
                          setEditingKey(null);
                          setEditSeverity("");
                          setEditRoute("");
                          setEditTimePickerOpen(false);
                          setEditRoutePickerOpen(false);
                        }}
                      >
                        <Text style={{ color: "#374064", fontFamily: "Exo2-SemiBold" }}>Cancel</Text>
                      </TouchableOpacity>
                      <TouchableOpacity
                        style={{ backgroundColor: "#FFEDEE", borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8 }}
                        disabled={busyKey === item.key}
                        onPress={async () => {
                          const confirmed = await confirmAction(
                            "Delete event",
                            "This will remove the event and trigger downstream recompute."
                          );
                          if (!confirmed) return;
                          setBusyKey(item.key);
                          const removedEventType = event.event_type;
                          const removedEventId = event.id;
                          setEvents((prev) =>
                            prev.filter((row) => !(row.event_type === removedEventType && row.id === removedEventId))
                          );
                          try {
                            if (!user) {
                              Alert.alert("Not authenticated", "Please sign in again.");
                              return;
                            }
                            await deleteEvent(removedEventType, removedEventId, user.id);
                            setEditingKey(null);
                            setEditSeverity("");
                            setEditRoute("");
                            setEditTimePickerOpen(false);
                            setEditRoutePickerOpen(false);
                            void load(false);
                          } catch {
                            void load(false);
                          } finally {
                            setBusyKey(null);
                          }
                        }}
                      >
                        <Text style={{ color: "#B42318", fontFamily: "Exo2-SemiBold" }}>Delete</Text>
                      </TouchableOpacity>
                    </View>
                  </View>
                ) : (
                  <View style={{ marginTop: 8, flexDirection: "row", gap: 10 }}>
                    <TouchableOpacity
                      style={{ backgroundColor: "#EEF3FF", borderRadius: 8, paddingHorizontal: 12, paddingVertical: 8 }}
                      onPress={() => {
                        setEditingKey(item.key);
                        setEditDate(eventDate(event.timestamp) ?? new Date());
                        setEditSeverity(event.severity != null ? String(event.severity) : "");
                        setEditRoute(normalizeRouteValue(event.route));
                      }}
                    >
                      <Text style={{ color: "#2E5BCE", fontFamily: "Exo2-SemiBold" }}>Edit</Text>
                    </TouchableOpacity>
                  </View>
                )}
              </View>
            </View>
          );
        }}
      />
      <Modal
        visible={selectedInsights.length > 0}
        transparent
        animationType="fade"
        onRequestClose={() => setSelectedInsights([])}
      >
        <View style={{ flex: 1, backgroundColor: "rgba(5,7,12,0.58)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "#FFFFFF", borderRadius: 14, borderWidth: 1, borderColor: "#E6E9F2", padding: 14, gap: 8, maxHeight: "80%" }}>
            <Text style={{ fontSize: 16, fontFamily: "Exo2-Bold", color: "#343A52" }}>
              {selectedInsights.length === 1 ? "Linked insight" : `Linked insights (${selectedInsights.length})`}
            </Text>
            <ScrollView showsVerticalScrollIndicator>
              {selectedInsights.map((selectedInsight) => (
                <View
                  key={`timeline-insight-${selectedInsight.id}`}
                  style={{ borderWidth: 1, borderColor: "#E6E9F2", borderRadius: 12, padding: 10, marginBottom: 8, gap: 4 }}
                >
                  {(() => {
                    const isVerified = Boolean(selectedInsight.user_verified);
                    const isRejected = Boolean(selectedInsight.user_rejected);
                    const showVerify = !isRejected;
                    const showReject = !isVerified;
                    return (
                      <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center", gap: 8 }}>
                        <Text style={{ fontSize: 15, fontFamily: "Exo2-Bold", color: "#343A52", flex: 1 }}>
                          {(selectedInsight.source_ingredient_name
                            ? `${selectedInsight.source_ingredient_name} (in ${selectedInsight.item_name})`
                            : selectedInsight.item_name)}{" "}
                          to {selectedInsight.symptom_name}
                        </Text>
                        <View style={{ flexDirection: "row", gap: 6 }}>
                          {showVerify ? (
                            <TouchableOpacity
                              disabled={verifyingInsightId === selectedInsight.id}
                              style={{
                                borderRadius: 999,
                                borderWidth: 1,
                                borderColor: isVerified ? "#2BAA6E" : "#2E5BCE",
                                backgroundColor: isVerified ? "#EEFCF3" : "#EEF3FF",
                                shadowColor: isVerified ? "#2BAA6E" : "#2E5BCE",
                                shadowOpacity: 0.12,
                                shadowRadius: 6,
                                shadowOffset: { width: 0, height: 2 },
                                paddingHorizontal: 8,
                                paddingVertical: 4,
                                opacity: verifyingInsightId === selectedInsight.id ? 0.6 : 1,
                              }}
                              onPress={() => {
                                const nextVerified = !isVerified;
                                Alert.alert(
                                  nextVerified ? "Verify insight?" : "Remove verification?",
                                  nextVerified
                                    ? "Confirm that this insight feels accurate for your experience?"
                                    : "Are you sure you want to remove your verification for this insight?",
                                  [
                                    { text: "Cancel", style: "cancel" },
                                    {
                                      text: nextVerified ? "Verify" : "Remove",
                                      style: nextVerified ? "default" : "destructive",
                                      onPress: async () => {
                                        setVerifyingInsightId(selectedInsight.id);
                                        try {
                                          if (!user) {
                                            Alert.alert("Not authenticated", "Please sign in again.");
                                            return;
                                          }
                                          await setInsightVerification(selectedInsight.id, user.id, nextVerified);
                                          setSelectedInsights((prev) =>
                                            prev.map((row) =>
                                              row.id === selectedInsight.id
                                                ? {
                                                    ...row,
                                                    user_verified: nextVerified,
                                                    user_rejected: nextVerified ? false : row.user_rejected,
                                                  }
                                                : row
                                            )
                                          );
                                          setInsightsById((prev) => {
                                            const current = prev[selectedInsight.id];
                                            if (!current) return prev;
                                            return {
                                              ...prev,
                                              [selectedInsight.id]: {
                                                ...current,
                                                user_verified: nextVerified,
                                                user_rejected: nextVerified ? false : current.user_rejected,
                                              },
                                            };
                                          });
                                          void load(false);
                                        } catch (err: any) {
                                          Alert.alert("Error", err?.message ?? "Failed to update verification.");
                                          void load(false);
                                        } finally {
                                          setVerifyingInsightId(null);
                                        }
                                      },
                                    },
                                  ]
                                );
                              }}
                            >
                              <Text
                                style={{
                                  color: isVerified ? "#1C8D57" : "#2E5BCE",
                                  fontFamily: "Exo2-SemiBold",
                                  fontSize: 10,
                                }}
                              >
                                {isVerified ? "Verified" : "Verify"}
                              </Text>
                            </TouchableOpacity>
                          ) : null}
                          {showReject ? (
                            <TouchableOpacity
                              disabled={verifyingInsightId === selectedInsight.id}
                              style={{
                                borderRadius: 999,
                                borderWidth: 1,
                                borderColor: isRejected ? "#C53F3F" : "#B54708",
                                backgroundColor: isRejected ? "#FEEEEE" : "#FFF4E8",
                                shadowColor: isRejected ? "#C53F3F" : "#B54708",
                                shadowOpacity: 0.12,
                                shadowRadius: 6,
                                shadowOffset: { width: 0, height: 2 },
                                paddingHorizontal: 8,
                                paddingVertical: 4,
                                opacity: verifyingInsightId === selectedInsight.id ? 0.6 : 1,
                              }}
                              onPress={() => {
                                const nextRejected = !isRejected;
                                Alert.alert(
                                  nextRejected ? "Reject insight?" : "Remove rejection?",
                                  nextRejected
                                    ? "Confirm that this insight does not match your experience?"
                                    : "Are you sure you want to remove your rejection for this insight?",
                                  [
                                    { text: "Cancel", style: "cancel" },
                                    {
                                      text: nextRejected ? "Reject" : "Remove",
                                      style: "destructive",
                                      onPress: async () => {
                                        setVerifyingInsightId(selectedInsight.id);
                                        try {
                                          if (!user) {
                                            Alert.alert("Not authenticated", "Please sign in again.");
                                            return;
                                          }
                                          await setInsightRejection(selectedInsight.id, user.id, nextRejected);
                                          setSelectedInsights((prev) =>
                                            prev.map((row) =>
                                              row.id === selectedInsight.id
                                                ? {
                                                    ...row,
                                                    user_rejected: nextRejected,
                                                    user_verified: nextRejected ? false : row.user_verified,
                                                  }
                                                : row
                                            )
                                          );
                                          setInsightsById((prev) => {
                                            const current = prev[selectedInsight.id];
                                            if (!current) return prev;
                                            return {
                                              ...prev,
                                              [selectedInsight.id]: {
                                                ...current,
                                                user_rejected: nextRejected,
                                                user_verified: nextRejected ? false : current.user_verified,
                                              },
                                            };
                                          });
                                          void load(false);
                                        } catch (err: any) {
                                          Alert.alert("Error", err?.message ?? "Failed to update rejection.");
                                          void load(false);
                                        } finally {
                                          setVerifyingInsightId(null);
                                        }
                                      },
                                    },
                                  ]
                                );
                              }}
                            >
                              <Text
                                style={{
                                  color: isRejected ? "#C53F3F" : "#B54708",
                                  fontFamily: "Exo2-SemiBold",
                                  fontSize: 10,
                                }}
                              >
                                {isRejected ? "Rejected" : "Reject"}
                              </Text>
                            </TouchableOpacity>
                          ) : null}
                        </View>
                      </View>
                    );
                  })()}
                  {(() => {
                    const firstCitation = (selectedInsight.citations ?? [])[0];
                    const formattedEvidence = formatEvidenceSummary(selectedInsight.evidence_summary, {
                      sourceLabel: firstCitation ? citationSourceLabel(firstCitation) : undefined,
                      citationTitle: firstCitation?.title ?? undefined,
                      abstractSnippet: firstCitation?.snippet ?? null,
                    });
                    return (
                      <>
                        <Text style={{ fontSize: 14, fontFamily: "Exo2-Regular", color: "#343A52" }}>
                          {formattedEvidence.summary}
                        </Text>
                        {formattedEvidence.onset ? (
                          <Text style={{ fontSize: 12, fontFamily: "Exo2-Regular", color: "#69708A" }}>
                            {formattedEvidence.onset}
                          </Text>
                        ) : null}
                      </>
                    );
                  })()}
                  <Text style={{ fontSize: 12, fontFamily: "Exo2-Regular", color: "#69708A" }}>
                    Confidence: {typeof selectedInsight.overall_confidence_score === "number" ? selectedInsight.overall_confidence_score.toFixed(2) : "-"}
                  </Text>
                  <Text style={{ fontSize: 12, fontFamily: "Exo2-Regular", color: "#69708A" }}>
                    Evidence strength: {typeof selectedInsight.evidence_strength_score === "number" ? selectedInsight.evidence_strength_score.toFixed(2) : "-"}
                  </Text>
                  <Text style={{ marginTop: 2, fontSize: 13, fontFamily: "Exo2-Bold", color: "#232A44" }}>
                    Citations ({selectedInsight.citations?.length ?? 0})
                  </Text>
                  {(selectedInsight.citations ?? []).slice(0, 3).map((citation, index) => (
                    <Text key={`timeline-insight-${selectedInsight.id}-${index}`} style={{ fontSize: 12, fontFamily: "Exo2-Regular", color: "#343A52" }}>
                      {citationSourceLabel(citation)}
                      {": "}
                      {citation.title ?? "Untitled"}
                    </Text>
                  ))}
                </View>
              ))}
            </ScrollView>
            <Pressable
              onPress={() => setSelectedInsights([])}
              style={{
                marginTop: 10,
                alignSelf: "flex-end",
                borderRadius: 8,
                backgroundColor: "#EEF3FF",
                paddingHorizontal: 14,
                paddingVertical: 8,
              }}
            >
              <Text style={{ color: "#2E5BCE", fontFamily: "Exo2-SemiBold" }}>Close</Text>
            </Pressable>
          </View>
        </View>
      </Modal>
      <Modal visible={editTimePickerOpen} transparent animationType="fade" onRequestClose={() => setEditTimePickerOpen(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(16,24,40,0.35)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "white", borderRadius: 14, padding: 14, gap: 10 }}>
            <Text style={{ fontSize: 16, fontFamily: "Exo2-Bold" }}>Edit Event Time</Text>
            {(
              [
                ["year", String(editDate.getFullYear())],
                ["month", String(editDate.getMonth() + 1).padStart(2, "0")],
                ["day", String(editDate.getDate()).padStart(2, "0")],
                ["hour", String(editDate.getHours()).padStart(2, "0")],
                ["minute", String(editDate.getMinutes()).padStart(2, "0")],
              ] as [("year" | "month" | "day" | "hour" | "minute"), string][]
            ).map(([part, value]) => (
              <View key={`timeline-edit-time-${part}`} style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "center" }}>
                <Text style={{ textTransform: "capitalize", color: "#343A52" }}>{part}</Text>
                <View style={{ flexDirection: "row", gap: 8, alignItems: "center" }}>
                  <Pressable
                    onPress={() => setEditDate((d) => shiftDatePart(d, part, -1))}
                    style={{ paddingHorizontal: 10, paddingVertical: 6, borderWidth: 1, borderColor: "#D0D5DD", borderRadius: 8 }}
                  >
                    <Text>-</Text>
                  </Pressable>
                  <Text style={{ minWidth: 56, textAlign: "center", fontFamily: "Exo2-SemiBold" }}>{value}</Text>
                  <Pressable
                    onPress={() => setEditDate((d) => shiftDatePart(d, part, 1))}
                    style={{ paddingHorizontal: 10, paddingVertical: 6, borderWidth: 1, borderColor: "#D0D5DD", borderRadius: 8 }}
                  >
                    <Text>+</Text>
                  </Pressable>
                </View>
              </View>
            ))}
            <Text style={{ color: "#343A52", marginTop: 2 }}>{formatIsoForDisplay(toIsoFromParts(editDate))}</Text>
            <View style={{ flexDirection: "row", gap: 10, marginTop: 6 }}>
              <View style={{ flex: 1 }}>
                <Button title="Done" onPress={() => setEditTimePickerOpen(false)} />
              </View>
              <View style={{ flex: 1 }}>
                <Button
                  title="Now"
                  onPress={() => {
                    setEditDate(new Date());
                    setEditTimePickerOpen(false);
                  }}
                />
              </View>
            </View>
          </View>
        </View>
      </Modal>
      <Modal visible={editRoutePickerOpen} transparent animationType="fade" onRequestClose={() => setEditRoutePickerOpen(false)}>
        <View style={{ flex: 1, backgroundColor: "rgba(16,24,40,0.35)", justifyContent: "center", padding: 18 }}>
          <View style={{ backgroundColor: "white", borderRadius: 14, padding: 14, gap: 10 }}>
            <Text style={{ fontSize: 16, fontFamily: "Exo2-Bold" }}>Edit Route</Text>
            <ScrollView style={{ maxHeight: 300 }}>
              {ROUTE_OPTIONS.map((option) => (
                <Pressable
                  key={`timeline-route-option-${option.value}`}
                  onPress={() => {
                    setEditRoute(option.value);
                    setEditRoutePickerOpen(false);
                  }}
                  style={{
                    borderWidth: 1,
                    borderColor: normalizeRouteValue(editRoute) === option.value ? "#2E5BCE" : "#D0D5DD",
                    borderRadius: 10,
                    paddingVertical: 10,
                    paddingHorizontal: 12,
                    backgroundColor: normalizeRouteValue(editRoute) === option.value ? "#EEF3FF" : "#FFF",
                    marginBottom: 8,
                  }}
                >
                  <Text
                    style={{
                      color: normalizeRouteValue(editRoute) === option.value ? "#2E5BCE" : "#343A52",
                      fontFamily: "Exo2-SemiBold",
                    }}
                  >
                    {option.label}
                  </Text>
                </Pressable>
              ))}
            </ScrollView>
            <View style={{ flexDirection: "row", gap: 10, marginTop: 4 }}>
              <View style={{ flex: 1 }}>
                <Button title="Done" onPress={() => setEditRoutePickerOpen(false)} />
              </View>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}
