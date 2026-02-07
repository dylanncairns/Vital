import React, { useState } from "react";
import { View, Text, TextInput, Button } from "react-native";
import { createEvent } from "../api/client";
import { ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

// Set to 1 for seed data temporarily
const USER_ID = 1;

// const [] items hold current value, useState has defaultvalue and states will update as user interacts
export default function LogEventScreen() {
    const [eventType, setEventType] = useState<"exposure" | "symptom">("exposure");
    const [timestamp, setTimestamp] = useState("");
    const [itemId, setItemId] = useState("");
    const [route, setRoute] = useState("");
    const [symptomId, setSymptomId] = useState("");
    const [severity, setSeverity] = useState("");
    const [error, setError] = useState<string | null>(null);


    // submit either exposure event or symptom event - run when user clicks "Log Event"
    async function submit() {
        setError(null);

        if (!timestamp.trim()) {
            setError("Timestamp is required.");
            return;
        }

        if (eventType === "exposure") {
            if (!itemId.trim() || !route.trim()) {
            setError("Exposure requires Item ID and Route.");
            return;
            }
            await createEvent({
                event_type: "exposure",
                user_id: USER_ID,
                timestamp,
                item_id: Number(itemId),
                route,
            });
        } else {
            if (!symptomId.trim()) {
                setError("Symptom requires Symptom ID.");
                return;
            }
            await createEvent({
                event_type: "symptom",
                user_id: USER_ID,
                timestamp,
                symptom_id: Number(symptomId),
                severity: severity ? Number(severity) : undefined,
            });
        }
    }

    // view
    return (
    <SafeAreaView style={{ flex: 1 }}>
      <ScrollView contentContainerStyle={{ padding: 16, gap: 14 }}>
        <Text style={{ fontSize: 18, fontWeight: "600" }}>Log Event</Text>

        <View>
          <Text style={{ marginBottom: 6 }}>Event Type (exposure or symptom)</Text>
          <TextInput
            style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
            value={eventType}
            onChangeText={(t) => setEventType(t as any)}
          />
        </View>

        {error ? <Text style={{ color: "red" }}>{error}</Text> : null}

        <View>
          <Text style={{ marginBottom: 6 }}>Timestamp (ISO)</Text>
          <TextInput
            style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
            value={timestamp}
            onChangeText={setTimestamp}
            placeholder="2026-02-07T12:00:00Z"
          />
        </View>

        {eventType === "exposure" ? (
          <>
            <View>
              <Text style={{ marginBottom: 6 }}>Item ID</Text>
              <TextInput
                style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
                value={itemId}
                onChangeText={setItemId}
                placeholder="1"
              />
            </View>
            <View>
              <Text style={{ marginBottom: 6 }}>Route</Text>
              <TextInput
                style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
                value={route}
                onChangeText={setRoute}
                placeholder="ingestion"
              />
            </View>
          </>
        ) : (
          <>
            <View>
              <Text style={{ marginBottom: 6 }}>Symptom ID</Text>
              <TextInput
                style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
                value={symptomId}
                onChangeText={setSymptomId}
                placeholder="1"
              />
            </View>
            <View>
              <Text style={{ marginBottom: 6 }}>Severity (1–5)</Text>
              <TextInput
                style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
                value={severity}
                onChangeText={setSeverity}
                placeholder="3"
              />
            </View>
          </>
        )}

        <Button title="Log Event" onPress={submit} />
      </ScrollView>
    </SafeAreaView>
  );
}