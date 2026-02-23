import React, { useEffect, useState } from "react";
import { Alert, FlatList, RefreshControl, Text, TouchableOpacity, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { useFocusEffect } from "@react-navigation/native";

import { fetchInsights, setInsightRejection, setInsightVerification } from "../api/client";
import { styles } from "./InsightsScreen.styles";
import { useAuth } from "../auth/AuthContext";
import { Insight } from "../models/events";

function prettyLagBucket(bucket: string): string {
  const map: Record<string, string> = {
    "0_6h": "0–6 hours",
    "6_24h": "6–24 hours",
    "24_72h": "24–72 hours",
    "72h_7d": "72 hours–7 days",
  };
  return map[bucket] ?? bucket;
}

function formatEvidenceSummary(
  summary: string | null | undefined,
  opts?: { sourceLabel?: string; citationTitle?: string; abstractSnippet?: string | null }
): { summary: string; onset: string | null } {
  let raw = (summary ?? "No summary").trim();
  // Remove redundant retrieval-count copy; citation count is already shown in UI.
  raw = raw
    .replace(/^\s*\d+\s+claim(?:s|\(s\))?\s+retrieved\s*[:;.\-]?\s*/i, "")
    .replace(/\b\d+\s+claim(?:s|\(s\))?\s+retrieved\s*[:;.\-]?\s*/gi, "")
    .replace(/\s{2,}/g, " ")
    .trim();

  const match = raw.match(/Dominant lag window:\s*([A-Za-z0-9_]+)\.?$/);
  let onset: string | null = null;
  if (match) {
    onset = `Most common onset in your data: ${prettyLagBucket(match[1])}`;
    raw = raw.replace(/Dominant lag window:\s*[A-Za-z0-9_]+\.?$/, "").trim();
  }

  const sourcePart = opts?.sourceLabel ? `from ${opts.sourceLabel} ` : "";
  const titlePart = opts?.citationTitle ? `(${opts.citationTitle}) ` : "";
  const snippetRaw = (opts?.abstractSnippet ?? "").replace(/\s+/g, " ").trim();
  const snippet = snippetRaw.length > 220 ? `${snippetRaw.slice(0, 217).trimEnd()}...` : snippetRaw;
  const snippetPart = snippet ? `${snippet} ` : "";
  const intro = snippet
    ? `Supportive evidence states that ${snippetPart}`
    : `Supportive evidence ${sourcePart}${titlePart}indicates that `;
  raw = raw.replace(/^overall evidence is supportive\s*(that)?\s*/i, intro);
  if (raw && !/[.!?]$/.test(raw)) {
    raw = `${raw}.`;
  }
  return { summary: raw || "No summary.", onset };
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

export default function InsightsScreen() {
  const { user } = useAuth();
  const [rows, setRows] = useState<Insight[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [pendingInsightId, setPendingInsightId] = useState<number | null>(null);

  async function confirmAction(
    title: string,
    message: string,
    confirmLabel: string = "Confirm",
    confirmStyle: "default" | "destructive" = "default"
  ): Promise<boolean> {
    if (typeof window !== "undefined" && typeof window.confirm === "function") {
      return window.confirm(`${title}\n\n${message}`);
    }
    return new Promise((resolve) => {
      Alert.alert(title, message, [
        { text: "Cancel", style: "cancel", onPress: () => resolve(false) },
        { text: confirmLabel, style: confirmStyle, onPress: () => resolve(true) },
      ]);
    });
  }

  const load = React.useCallback(async () => {
    setRefreshing(true);
    try {
      if (!user) return;
      const data = await fetchInsights(user.id, false);
      setRows(data);
    } finally {
      setRefreshing(false);
    }
  }, [user]);

  useEffect(() => {
    load();
  }, [load]);

  useFocusEffect(
    React.useCallback(() => {
      load();
    }, [load])
  );

  return (
    <SafeAreaView style={styles.safe}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>Insights</Text>
      </View>
      <FlatList
        data={rows}
        keyExtractor={(item) => `${item.id}`}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerStyle={[styles.container, { flexGrow: 1, paddingTop: 6 }]}
        ListEmptyComponent={<Text style={styles.emptyText}>Add more events to generate insights.</Text>}
        ListFooterComponent={
          <Text
            style={{
              marginTop: 10,
              marginBottom: 0,
              textAlign: "center",
              fontSize: 12,
              fontFamily: "Exo2-Regular",
              color: "#8C92A6",
            }}
          >
            Vital does not intend to and does not have liscencing to give medical advice.
          </Text>
        }
        renderItem={({ item }) => {
          const isVerified = Boolean(item.user_verified);
          const isRejected = Boolean(item.user_rejected);
          const showVerify = !isRejected;
          const showReject = !isVerified;
          const firstCitation = item.citations?.[0];
          const formattedEvidence = formatEvidenceSummary(item.evidence_summary, {
            sourceLabel: firstCitation ? citationSourceLabel(firstCitation) : undefined,
            citationTitle: firstCitation?.title ?? undefined,
            abstractSnippet: firstCitation?.snippet ?? null,
          });
          return (
            <View style={styles.card}>
              <View style={{ flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start", gap: 8 }}>
                <Text style={[styles.title, { flex: 1 }]}>
                  {item.source_ingredient_name
                    ? `${item.source_ingredient_name} (in ${item.item_name})`
                    : item.item_name}
                  {" to "}
                  {item.symptom_name}
                </Text>
                <View style={{ flexDirection: "row", gap: 6 }}>
                  {showVerify ? (
                    <TouchableOpacity
                      disabled={pendingInsightId === item.id}
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
                        opacity: pendingInsightId === item.id ? 0.6 : 1,
                      }}
                      onPress={async () => {
                        const nextVerified = !isVerified;
                        const confirmed = await confirmAction(
                          nextVerified ? "Verify insight?" : "Remove verification?",
                          nextVerified
                            ? "Confirm that this insight feels accurate for your experience?"
                            : "Are you sure you want to remove your verification for this insight?",
                          nextVerified ? "Verify" : "Remove",
                          nextVerified ? "default" : "destructive"
                        );
                        if (!confirmed) return;
                        if (!user) return;
                        setPendingInsightId(item.id);
                        try {
                          await setInsightVerification(item.id, user.id, nextVerified);
                          setRows((prev) =>
                            prev.map((row) =>
                              row.id === item.id
                                ? {
                                    ...row,
                                    user_verified: nextVerified,
                                    user_rejected: nextVerified ? false : row.user_rejected,
                                  }
                                : row
                            )
                          );
                        } catch (err: any) {
                          Alert.alert("Error", err?.message ?? "Failed to update verification.");
                        } finally {
                          setPendingInsightId(null);
                        }
                      }}
                    >
                      <Text
                        style={{
                          color: isVerified ? "#1C8D57" : "#2E5BCE",
                          fontFamily: "Exo2-SemiBold",
                          fontSize: 11,
                        }}
                      >
                        {isVerified ? "Verified" : "Verify insight"}
                      </Text>
                    </TouchableOpacity>
                  ) : null}
                  {showReject ? (
                    <TouchableOpacity
                      disabled={pendingInsightId === item.id}
                      style={{
                        borderRadius: 999,
                        borderWidth: 1,
                        borderColor: isRejected ? "#E77A7A" : "#F6B889",
                        backgroundColor: isRejected ? "#FEEEEE" : "#FFF4E8",
                        shadowColor: isRejected ? "#D25A5A" : "#E58D3D",
                        shadowOpacity: 0.1,
                        shadowRadius: 6,
                        shadowOffset: { width: 0, height: 2 },
                        paddingHorizontal: 8,
                        paddingVertical: 4,
                        opacity: pendingInsightId === item.id ? 0.6 : 1,
                      }}
                      onPress={async () => {
                        const nextRejected = !isRejected;
                        const confirmed = await confirmAction(
                          nextRejected ? "Reject insight?" : "Remove rejection?",
                          nextRejected
                            ? "Mark that this insight does not fit your experience?"
                            : "Are you sure you want to remove your rejection for this insight?",
                          nextRejected ? "Reject" : "Remove",
                          "destructive"
                        );
                        if (!confirmed) return;
                        if (!user) return;
                        setPendingInsightId(item.id);
                        try {
                          await setInsightRejection(item.id, user.id, nextRejected);
                          setRows((prev) =>
                            prev.map((row) =>
                              row.id === item.id
                                ? {
                                    ...row,
                                    user_rejected: nextRejected,
                                    user_verified: nextRejected ? false : row.user_verified,
                                  }
                                : row
                            )
                          );
                        } catch (err: any) {
                          Alert.alert("Error", err?.message ?? "Failed to update rejection.");
                        } finally {
                          setPendingInsightId(null);
                        }
                      }}
                    >
                      <Text
                        style={{
                          color: isRejected ? "#C53F3F" : "#B54708",
                          fontFamily: "Exo2-SemiBold",
                          fontSize: 11,
                        }}
                      >
                        {isRejected ? "Rejected" : "Reject insight"}
                      </Text>
                    </TouchableOpacity>
                  ) : null}
                </View>
              </View>
              <Text style={styles.summary}>{formattedEvidence.summary}</Text>
              {formattedEvidence.onset ? <Text style={styles.meta}>{formattedEvidence.onset}</Text> : null}
              <Text style={styles.meta}>
                Confidence: {typeof item.overall_confidence_score === "number" ? item.overall_confidence_score.toFixed(2) : "-"}
              </Text>
              <Text style={styles.meta}>
                Evidence strength: {typeof item.evidence_strength_score === "number" ? item.evidence_strength_score.toFixed(2) : "-"}
              </Text>
              <Text style={styles.citationsHeader}>Citations ({item.citations.length})</Text>
              {item.citations.slice(0, 3).map((citation, index) => (
                <Text key={`${item.id}-citation-${index}`} style={styles.citationLine}>
                  {citationSourceLabel(citation)}
                  {": "}
                  {citation.title ?? "Untitled"}
                </Text>
              ))}
            </View>
          );
        }}
      />
    </SafeAreaView>
  );
}
