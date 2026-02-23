import React, { useMemo, useState } from "react";
import {
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect } from "@react-navigation/native";

import { fetchEvents, fetchInsights } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { Insight, TimelineEvent } from "../models/events";
import { styles as sharedStyles } from "./InsightsScreen.styles";

type InsightGroupSummary = {
  name: string;
  count: number;
  avgConfidence: number | null;
  insights: Insight[];
};

function insightLabel(insight: Insight): string {
  const exposure = insight.source_ingredient_name
    ? `${insight.source_ingredient_name} (in ${insight.item_name})`
    : insight.item_name;
  return `${exposure} to ${insight.symptom_name}`;
}

function formatConfidence(score: number | null): string {
  if (score == null || !Number.isFinite(score)) return "N/A";
  return score.toFixed(2);
}

function averageConfidence(insights: Insight[]): number | null {
  const values = insights
    .map((row) => row.overall_confidence_score)
    .filter((value): value is number => typeof value === "number" && Number.isFinite(value));
  if (!values.length) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function prettyLagBucket(bucket: string): string {
  const map: Record<string, string> = {
    "0_6h": "0–6 hours",
    "6_24h": "6–24 hours",
    "24_72h": "24–72 hours",
    "72h_7d": "72 hours–7 days",
  };
  return map[bucket] ?? bucket.replace(/_/g, " - ");
}

function formatEvidenceSummary(
  summary: string | null | undefined
): { summary: string; onset: string | null } {
  let raw = (summary || "").trim();
  if (!raw) return { summary: "No evidence summary available yet.", onset: null };

  const match = raw.match(/Dominant lag window:\s*([A-Za-z0-9_]+)\.?$/i);
  let onset: string | null = null;
  if (match) {
    onset = `Most common onset in your data: ${prettyLagBucket(match[1])}`;
    raw = raw.replace(/Dominant lag window:\s*[A-Za-z0-9_]+\.?$/i, "").trim();
  }
  if (raw && !/[.!?]$/.test(raw)) {
    raw = `${raw}.`;
  }
  return { summary: raw || "No evidence summary available yet.", onset };
}

function citationSourceLabel(citation: { source?: string | null; url?: string | null }): string {
  const source = (citation.source ?? "").trim();
  const sourceLower = source.toLowerCase();
  if (source && sourceLower !== "openai_file_search" && sourceLower !== "file_search") {
    return source;
  }
  const url = (citation.url ?? "").trim();
  if (!url) return "Unknown Source";
  try {
    return new URL(url).hostname || "Unknown Source";
  } catch {
    return "Unknown Source";
  }
}

function summarizeByExposure(insights: Insight[]): InsightGroupSummary | null {
  const grouped = new Map<string, Insight[]>();
  for (const row of insights) {
    const key = (row.item_name || "").trim();
    if (!key) continue;
    const existing = grouped.get(key) ?? [];
    existing.push(row);
    grouped.set(key, existing);
  }
  let best: [string, Insight[]] | null = null;
  for (const entry of grouped.entries()) {
    if (!best || entry[1].length > best[1].length) best = entry;
  }
  if (!best) return null;
  return {
    name: best[0],
    count: best[1].length,
    avgConfidence: averageConfidence(best[1]),
    insights: best[1],
  };
}

function summarizeByMostReportedSymptom(events: TimelineEvent[], insights: Insight[]): InsightGroupSummary | null {
  const symptomCounts = new Map<string, number>();
  for (const row of events) {
    if (row.event_type !== "symptom") continue;
    const name = (row.symptom_name || "").trim();
    if (!name) continue;
    symptomCounts.set(name, (symptomCounts.get(name) ?? 0) + 1);
  }
  let mostReportedName = "";
  let mostReportedCount = 0;
  for (const [name, count] of symptomCounts.entries()) {
    if (count > mostReportedCount) {
      mostReportedName = name;
      mostReportedCount = count;
    }
  }
  if (!mostReportedName) return null;
  const matchingInsights = insights.filter(
    (row) => (row.symptom_name || "").trim().toLowerCase() === mostReportedName.toLowerCase()
  );
  return {
    name: mostReportedName,
    count: mostReportedCount,
    avgConfidence: averageConfidence(matchingInsights),
    insights: matchingInsights,
  };
}

function InsightAccordionList({
  rows,
  selectedInsightId,
  onToggle,
}: {
  rows: Insight[];
  selectedInsightId: number | null;
  onToggle: (id: number) => void;
}) {
  if (!rows.length) {
    return <Text style={localStyles.emptyInline}>No surfaced insights yet for this entry.</Text>;
  }

  const selectedInsight = rows.find((row) => row.id === selectedInsightId) ?? null;
  const selectedFormatted = selectedInsight ? formatEvidenceSummary(selectedInsight.evidence_summary) : null;

  return (
    <View style={localStyles.listPanelRow}>
      <View style={localStyles.listColumn}>
        {rows.map((row) => {
          const open = selectedInsightId === row.id;
          return (
            <View key={`your-data-insight-${row.id}`} style={localStyles.accordionRow}>
              <TouchableOpacity onPress={() => onToggle(row.id)} style={localStyles.accordionButton}>
                <Text numberOfLines={1} ellipsizeMode="tail" style={localStyles.accordionTitle}>
                  {insightLabel(row)}
                </Text>
                <Text style={localStyles.accordionChevron}>{open ? "Details -" : "Details"}</Text>
              </TouchableOpacity>
            </View>
          );
        })}
      </View>
      <View style={localStyles.panelColumn}>
        {selectedInsight && selectedFormatted ? (
          <View style={localStyles.detailPanel}>
            <View style={localStyles.detailHeaderRow}>
              <Text style={[localStyles.detailPanelTitle, { flex: 1 }]}>{insightLabel(selectedInsight)}</Text>
              <View style={{ flexDirection: "row", gap: 6, flexWrap: "wrap" }}>
                {selectedInsight.user_verified ? (
                  <View style={[localStyles.statusPill, { borderColor: "#2BAA6E", backgroundColor: "#EEFCF3" }]}>
                    <Text style={[localStyles.statusPillText, { color: "#1C8D57" }]}>Verified</Text>
                  </View>
                ) : null}
                {selectedInsight.user_rejected ? (
                  <View style={[localStyles.statusPill, { borderColor: "#C53F3F", backgroundColor: "#FEEEEE" }]}>
                    <Text style={[localStyles.statusPillText, { color: "#C53F3F" }]}>Rejected</Text>
                  </View>
                ) : null}
              </View>
            </View>
            <Text style={localStyles.detailBodyText}>{selectedFormatted.summary}</Text>
            {selectedFormatted.onset ? <Text style={localStyles.dropdownMeta}>{selectedFormatted.onset}</Text> : null}
            <Text style={localStyles.dropdownMeta}>
              Confidence: {formatConfidence(selectedInsight.overall_confidence_score ?? null)}
            </Text>
            <Text style={localStyles.dropdownMeta}>
              Evidence strength:{" "}
              {typeof selectedInsight.evidence_strength_score === "number"
                ? selectedInsight.evidence_strength_score.toFixed(2)
                : "-"}
            </Text>
            <Text style={localStyles.citationsHeader}>
              Citations ({selectedInsight.citations?.length ?? 0})
            </Text>
            {(selectedInsight.citations ?? []).slice(0, 3).map((citation, index) => (
              <Text key={`your-data-citation-${selectedInsight.id}-${index}`} style={localStyles.citationLine}>
                {citationSourceLabel(citation)}: {citation.title ?? "Untitled"}
              </Text>
            ))}
          </View>
        ) : (
          <View style={[localStyles.detailPanel, localStyles.detailPanelPlaceholder]}>
            <Text style={localStyles.detailPanelTitle}>Insight details</Text>
            <Text style={localStyles.dropdownMeta}>Select an insight.</Text>
          </View>
        )}
      </View>
    </View>
  );
}

export default function YourDataScreen() {
  const { user } = useAuth();
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  const [insights, setInsights] = useState<Insight[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedExposureInsightId, setSelectedExposureInsightId] = useState<number | null>(null);
  const [selectedSymptomInsightId, setSelectedSymptomInsightId] = useState<number | null>(null);

  const load = React.useCallback(async () => {
    if (!user) return;
    setRefreshing(true);
    try {
      const [eventRows, insightRows] = await Promise.all([
        fetchEvents(user.id),
        fetchInsights(user.id, false),
      ]);
      setEvents(eventRows);
      setInsights(insightRows);
    } finally {
      setRefreshing(false);
    }
  }, [user]);

  React.useEffect(() => {
    void load();
  }, [load]);

  useFocusEffect(
    React.useCallback(() => {
      void load();
    }, [load])
  );

  const exposureSummary = useMemo(() => summarizeByExposure(insights), [insights]);
  const symptomSummary = useMemo(
    () => summarizeByMostReportedSymptom(events, insights),
    [events, insights]
  );

  return (
    <SafeAreaView style={sharedStyles.safe}>
      <View style={sharedStyles.header}>
        <Text style={sharedStyles.headerTitle}>Your Data</Text>
      </View>
      <ScrollView
        contentContainerStyle={[sharedStyles.container, { flexGrow: 1, paddingTop: 6 }]}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={() => void load()} />}
      >
        <View style={sharedStyles.card}>
          <Text style={localStyles.sectionHeader}>Most Flagged Exposure</Text>
          {exposureSummary ? (
            <>
              <Text style={sharedStyles.summary}>
                Your exposure to {exposureSummary.name} was flagged {exposureSummary.count}{" "}
                {exposureSummary.count === 1 ? "time" : "times"}.
              </Text>
              <Text style={sharedStyles.summary}>
                Our models compute an average confidence score of{" "}
                {formatConfidence(exposureSummary.avgConfidence)} for insights involving{" "}
                {exposureSummary.name}.
              </Text>
              <View style={localStyles.insightListSpacer}>
                <InsightAccordionList
                  rows={exposureSummary.insights}
                  selectedInsightId={selectedExposureInsightId}
                  onToggle={(id) =>
                    setSelectedExposureInsightId((prev) => (prev === id ? null : id))
                  }
                />
              </View>
            </>
          ) : (
            <Text style={sharedStyles.emptyText}>No surfaced exposure insights yet.</Text>
          )}
        </View>

        <View style={sharedStyles.card}>
          <Text style={localStyles.sectionHeader}>Most Flagged Symptom</Text>
          {symptomSummary ? (
            <>
              <Text style={sharedStyles.summary}>
                Your most reported symptom was {symptomSummary.name}, reported {symptomSummary.count}{" "}
                {symptomSummary.count === 1 ? "time" : "times"}.
              </Text>
              <Text style={sharedStyles.summary}>
                Our models compute an average confidence score of{" "}
                {formatConfidence(symptomSummary.avgConfidence)} for insights involving{" "}
                {symptomSummary.name}.
              </Text>
              <View style={localStyles.insightListSpacer}>
                <InsightAccordionList
                  rows={symptomSummary.insights}
                  selectedInsightId={selectedSymptomInsightId}
                  onToggle={(id) =>
                    setSelectedSymptomInsightId((prev) => (prev === id ? null : id))
                  }
                />
              </View>
            </>
          ) : (
            <Text style={sharedStyles.emptyText}>Log more symptom events to populate this section.</Text>
          )}
        </View>

        <Text style={localStyles.disclaimerText}>
          Vital does not intend to and does not have liscencing to give medical advice.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const localStyles = StyleSheet.create({
  sectionHeader: {
    fontSize: 18,
    fontFamily: "Exo2-Bold",
    color: "#101426",
    marginBottom: 2,
  },
  accordionRow: {
    borderWidth: 1,
    borderColor: "#E6E9F2",
    borderRadius: 10,
    backgroundColor: "#FBFCFF",
    alignSelf: "flex-start",
  },
  accordionButton: {
    paddingHorizontal: 10,
    paddingVertical: 10,
    flexDirection: "row",
    justifyContent: "flex-start",
    alignItems: "center",
    gap: 8,
  },
  accordionTitle: {
    width: 220,
    fontSize: 13,
    fontFamily: "Exo2-SemiBold",
    color: "#232A44",
  },
  accordionChevron: {
    fontSize: 12,
    fontFamily: "Exo2-SemiBold",
    color: "#2E5BCE",
    marginLeft: 16,
    width: 72,
    textAlign: "left",
  },
  listPanelRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    gap: 20,
  },
  listColumn: {
    flex: 1,
    minWidth: 0,
    gap: 8,
    alignItems: "flex-start",
  },
  panelColumn: {
    width: "38%",
    minWidth: 0,
  },
  insightListSpacer: {
    marginTop: 10,
  },
  detailPanel: {
    width: "100%",
    minHeight: 240,
    backgroundColor: "#FFFFFF",
    borderRadius: 14,
    borderWidth: 1,
    borderColor: "#E6E9F2",
    padding: 14,
    gap: 2,
    shadowColor: "#000",
    shadowOpacity: 0.05,
    shadowRadius: 6,
    shadowOffset: { width: 0, height: 2 },
  },
  detailPanelPlaceholder: {
    justifyContent: "flex-start",
  },
  detailPanelTitle: {
    fontSize: 14,
    fontFamily: "Exo2-Bold",
    color: "#343A52",
  },
  detailHeaderRow: {
    flexDirection: "row",
    alignItems: "flex-start",
    justifyContent: "space-between",
    gap: 8,
  },
  detailBodyText: {
    fontSize: 14,
    fontFamily: "Exo2-Regular",
    color: "#343A52",
  },
  statusPill: {
    borderRadius: 999,
    borderWidth: 1,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  statusPillText: {
    fontSize: 10,
    fontFamily: "Exo2-SemiBold",
  },
  citationsHeader: {
    marginTop: 3,
    fontSize: 13,
    fontFamily: "Exo2-Bold",
    color: "#232A44",
  },
  citationLine: {
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#343A52",
  },
  dropdownMeta: {
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#4D5674",
  },
  emptyInline: {
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#69708A",
  },
  disclaimerText: {
    marginTop: 10,
    marginBottom: 0,
    textAlign: "center",
    fontSize: 12,
    fontFamily: "Exo2-Regular",
    color: "#8C92A6",
  },
});
