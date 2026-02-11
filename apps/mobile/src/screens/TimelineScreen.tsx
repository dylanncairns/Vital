import React, { useEffect, useMemo, useState } from "react";
import { View, Text, FlatList, RefreshControl } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { fetchEvents } from "../api/client";
import { TimelineEvent } from "../models/events";
import { styles } from "./TimelineScreen.styles";

// Set to 1 for seed data temporarily
const USER_ID = 1;
type TimelineRow =
  | { type: "date"; key: string; label: string }
  | { type: "event"; key: string; event: TimelineEvent };

// Display list on screen in evenets entity and show refresh spinner when refreshing
export default function TimelineScreen() {
    const [events, setEvents] = useState<TimelineEvent[]>([]);
    const [refreshing, setRefreshing] = useState(false);

    function eventDate(value: string | null | undefined): Date | null {
      if (!value) return null;
      const parsed = new Date(value);
      return Number.isNaN(parsed.getTime()) ? null : parsed;
    }

    async function load() {
        setRefreshing(true);
        try {
            const data = await fetchEvents(USER_ID);
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
            setEvents(sorted);
        } finally {
            setRefreshing(false);
        }
    }

    // Runs when screen first appears - triggers first API fetch to populate timeline
    useEffect(() => {
        load();
    }, []);

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
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={load} />}
        contentContainerStyle={[styles.container, { flexGrow: 1 }]}
        bounces={true}
        alwaysBounceVertical={true}
        ListHeaderComponent={
          <View style={styles.header}>
            <Text style={styles.headerTitle}>Timeline</Text>
          </View>
        }
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
                <Text style={styles.titleText}>
                  {isExposure
                    ? `Exposure: ${event.item_name ?? "Item"}`
                    : `Symptom: ${event.symptom_name ?? "Symptom"}`}
                </Text>
                <Text style={styles.bodyText}>
                  {isExposure
                    ? `Route: ${event.route ?? "unknown"}`
                    : `Severity: ${event.severity ?? "-"}`}
                </Text>
              </View>
            </View>
          );
        }}
      />
    </SafeAreaView>
  );
}
