import React, { useEffect, useMemo, useState } from "react";
import { View, Text, FlatList, RefreshControl } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { fetchEvents } from "../api/client";
import { TimelineEvent } from "../models/events";
import { styles } from "./TimelineScreen.styles";

// Set to 1 for seed data temporarily
const USER_ID = 1;

// Display list on screen in evenets entity and show refresh spinner when refreshing
export default function TimelineScreen() {
    const [events, setEvents] = useState<TimelineEvent[]>([]);
    const [refreshing, setRefreshing] = useState(false);

    async function load() {
        setRefreshing(true);
        try {
            const data = await fetchEvents(USER_ID);
            setEvents(data);
        } finally {
            setRefreshing(false);
        }
    }

    // Runs when screen first appears - triggers first API fetch to populate timeline
    useEffect(() => {
        load();
    }, []);

    // Display of timeline + events
    const headerDate = useMemo(() => {
    if (!events.length) return "";
    const d = new Date(events[0].timestamp);
    return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
  }, [events]);

  return (
    <SafeAreaView style={styles.safe}>
      <FlatList
        data={events}
        keyExtractor={(item) => `${item.event_type}-${item.id}`}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerStyle={[styles.container, { flexGrow: 1 }]}
        bounces={true}
        alwaysBounceVertical={true}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Today</Text>
            <View style={styles.headerLine} />
            <Text style={styles.headerDate}>{headerDate}</Text>
          </View>
        }
        renderItem={({ item }) => {
          const isExposure = item.event_type === "exposure";
          const dotColor = isExposure ? "#00C389" : "#4F7BFF";
          const time = new Date(item.timestamp).toLocaleTimeString([], {
            hour: "2-digit",
            minute: "2-digit",
          });
          return (
            <View style={styles.row}>
              <View style={styles.railWrap}>
                <View style={styles.rail} />
                <View style={[styles.dot, { backgroundColor: dotColor }]} />
              </View>

              <View style={styles.card}>
                <Text style={styles.timeText}>{time}</Text>
                <Text style={styles.titleText}>
                  {isExposure
                    ? `Exposure: ${item.item_name ?? "Item"}`
                    : `Symptom: ${item.symptom_name ?? "Symptom"}`}
                </Text>
                <Text style={styles.bodyText}>
                  {isExposure
                    ? `Route: ${item.route ?? "unknown"}`
                    : `Severity: ${item.severity ?? 0}`}
                </Text>
                <Text style={styles.metaText}>
                  {isExposure
                    ? `Item ID: ${item.item_id ?? "-"}`
                    : `Symptom ID: ${item.symptom_id ?? "-"}`}
                </Text>
              </View>
            </View>
          );
        }}
      />
    </SafeAreaView>
  );
}