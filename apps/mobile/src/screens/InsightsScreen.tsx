import React, { useEffect, useState } from "react";
import { FlatList, RefreshControl, Text, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { fetchInsights } from "../api/client";
import { styles } from "./InsightsScreen.styles";

const USER_ID = 1;

function badgeForStatus(status: string | null | undefined) {
  if (status === "supported") {
    return { label: "Supported", bg: "#D7F5E8", fg: "#0A7A4F" };
  }
  if (status === "insufficient_evidence") {
    return { label: "Insufficient Evidence", bg: "#FFF0D4", fg: "#8A5C00" };
  }
  return { label: "Suppressed", bg: "#E9EDF6", fg: "#4C5470" };
}

export default function InsightsScreen() {
  const [rows, setRows] = useState([]);
  const [refreshing, setRefreshing] = useState(false);

  async function load() {
    setRefreshing(true);
    try {
      const data = await fetchInsights(USER_ID, false);
      setRows(data);
    } finally {
      setRefreshing(false);
    }
  }

  useEffect(() => {
    load();
  }, []);

  return (
    <SafeAreaView style={styles.safe}>
      <FlatList
        data={rows}
        keyExtractor={(item) => `${item.id}`}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerStyle={[styles.container, { flexGrow: 1 }]}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Insights</Text>
          </View>
        }
        ListEmptyComponent={<Text style={styles.emptyText}>No insights yet.</Text>}
        renderItem={({ item }) => {
          const badge = badgeForStatus(item.display_status ?? "suppressed");
          return (
            <View style={styles.card}>
              <View style={[styles.badge, { backgroundColor: badge.bg }]}>
                <Text style={[styles.badgeText, { color: badge.fg }]}>{badge.label}</Text>
              </View>
              <Text style={styles.title}>
                {item.item_name}
                {" to "}
                {item.symptom_name}
              </Text>
              <Text style={styles.summary}>{item.evidence_summary ?? "No summary"}</Text>
              <Text style={styles.meta}>
                Evidence strength: {typeof item.evidence_strength_score === "number" ? item.evidence_strength_score.toFixed(2) : "-"}
              </Text>
              <Text style={styles.meta}>Reason: {item.display_decision_reason ?? "-"}</Text>
              <Text style={styles.citationsHeader}>Citations ({item.citations.length})</Text>
              {item.citations.slice(0, 3).map((citation, index) => (
                <Text key={`${item.id}-citation-${index}`} style={styles.citationLine}>
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
