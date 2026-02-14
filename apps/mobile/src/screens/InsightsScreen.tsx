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

function formatEvidenceSummary(summary: string | null | undefined): string {
  const raw = (summary ?? "No summary").trim();
  const match = raw.match(/Dominant lag window:\s*([A-Za-z0-9_]+)\.?$/);
  if (!match) {
    return raw;
  }
  const bucket = match[1];
  const withoutLagSentence = raw.replace(/Dominant lag window:\s*[A-Za-z0-9_]+\.?$/, "").trim();
  const onsetText = `Most common onset in your data: ${prettyLagBucket(bucket)}`;
  return withoutLagSentence ? `${withoutLagSentence} ${onsetText}` : onsetText;
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
  const { user, logout } = useAuth();
  const [rows, setRows] = useState<Insight[]>([]);
  const [refreshing, setRefreshing] = useState(false);
  const [pendingInsightId, setPendingInsightId] = useState<number | null>(null);

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
      <FlatList
        data={rows}
        keyExtractor={(item) => `${item.id}`}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerStyle={[styles.container, { flexGrow: 1 }]}
        ListHeaderComponent={
          <View style={[styles.header, { flexDirection: "row", justifyContent: "space-between", alignItems: "center" }]}>
            <Text style={styles.headerTitle}>Insights</Text>
            <Text
              onPress={async () => {
                await logout();
              }}
              style={{ color: "#2E5BCE", fontFamily: "Exo2-SemiBold", fontSize: 13 }}
            >
              Logout
            </Text>
          </View>
        }
        ListEmptyComponent={<Text style={styles.emptyText}>No insights yet.</Text>}
        renderItem={({ item }) => {
          const isVerified = Boolean(item.user_verified);
          const isRejected = Boolean(item.user_rejected);
          const showVerify = !isRejected;
          const showReject = !isVerified;
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
                                if (!user) return;
                                setPendingInsightId(item.id);
                                try {
                                  await setInsightVerification(item.id, user.id, nextVerified);
                                  if (nextVerified) {
                                    await setInsightRejection(item.id, user.id, false);
                                  }
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
                      onPress={() => {
                        const nextRejected = !isRejected;
                        Alert.alert(
                          nextRejected ? "Reject insight?" : "Remove rejection?",
                          nextRejected
                            ? "Mark that this insight does not fit your experience?"
                            : "Are you sure you want to remove your rejection for this insight?",
                          [
                            { text: "Cancel", style: "cancel" },
                            {
                              text: nextRejected ? "Reject" : "Remove",
                              style: "destructive",
                              onPress: async () => {
                                if (!user) return;
                                setPendingInsightId(item.id);
                                try {
                                  await setInsightRejection(item.id, user.id, nextRejected);
                                  if (nextRejected) {
                                    await setInsightVerification(item.id, user.id, false);
                                  }
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
                          fontSize: 11,
                        }}
                      >
                        {isRejected ? "Rejected" : "Reject insight"}
                      </Text>
                    </TouchableOpacity>
                  ) : null}
                </View>
              </View>
              <Text style={styles.summary}>{formatEvidenceSummary(item.evidence_summary)}</Text>
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
