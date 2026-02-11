import React, { useState } from "react";
import { View, Text, TextInput, Button } from "react-native";
import { createEvent, ingestTextEvent } from "../api/client";
import { ScrollView } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

// Set to 1 for seed data temporarily
const USER_ID = 1;

// const [] items hold current value, useState has defaultvalue and states will update as user interacts
export default function LogEventScreen() {
    const [eventType, setEventType] = useState<"exposure" | "symptom">("exposure");
    const [timestamp, setTimestamp] = useState("");
    const [itemName, setItemName] = useState("");
    const [route, setRoute] = useState("");
    const [symptomId, setSymptomId] = useState("");
    const [severity, setSeverity] = useState("");
    const [error, setError] = useState<string | null>(null);
    const [rawText, setRawText] = useState("");
    const [textStatus, setTextStatus] = useState<string | null>(null);
    const [eventStatus, setEventStatus] = useState<string | null>(null);


    // submit either exposure event or symptom event - run when user clicks "Log Event"
    async function submit() {
        setError(null);
        setTextStatus(null);
        setEventStatus(null);

        if (!timestamp.trim()) {
            setError("Timestamp is required.");
            return;
        }
        try {
          if (eventType === "exposure") {
              if (!itemName.trim() || !route.trim()) {
              setError("Exposure requires Item Name and Route.");
              return;
              }
              const res = await createEvent({
                  event_type: "exposure",
                  user_id: USER_ID,
                  timestamp,
                  item_name: itemName,
                  route,
              });
              if (res.status === "queued") {
                setEventStatus(`Queued: ${res.resolution ?? "pending"}`);
              } else {
                setEventStatus("Event logged.");
              }
          } else {
              if (!symptomId.trim()) {
                  setError("Symptom requires Symptom ID.");
                  return;
              }
              const res = await createEvent({
                  event_type: "symptom",
                  user_id: USER_ID,
                  timestamp,
                  symptom_id: Number(symptomId),
                  severity: severity ? Number(severity) : undefined,
              });
              if (res.status === "queued") {
                setEventStatus(`Queued: ${res.resolution ?? "pending"}`);
              } else {
                setEventStatus("Event logged.");
              }
          }
        } catch (err: any) {
          setError(err.message ?? "Failed to create event.");
        }
    }
    // text blurb input - shows returned ingestion status
    async function submitText() {
        setError(null);
        setTextStatus(null);

        if (!rawText.trim()) {
            setError("Text input is required.");
            return;
        }
        try {
            const res = await ingestTextEvent({ user_id: USER_ID, raw_text: rawText });
            setTextStatus(`Text ingested: ${res.status}`);
            setRawText("");
        } catch (err: any) {
            setError(err.message ?? "Failed to ingest text.");
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
              <Text style={{ marginBottom: 6 }}>Item Name</Text>
              <TextInput
                style={{ borderWidth: 1, borderColor: "#ccc", padding: 10, borderRadius: 8 }}
                value={itemName}
                onChangeText={setItemName}
                placeholder="Rice"
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
        {eventStatus ? <Text style={{ marginTop: 8, color: "green" }}>{eventStatus}</Text> : null}

        <View style={{ marginTop: 18 }}>
          <Text style={{ fontSize: 16, fontWeight: "600", marginBottom: 6 }}>Quick Log (Text)</Text>
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
            placeholder="e.g. Ate sugar at 3pm. Headache started last night (severity 3)"
            multiline
          />
          <View style={{ marginTop: 10 }}>
            <Button title="Log Text" onPress={submitText} />
          </View>
          {textStatus ? <Text style={{ marginTop: 8, color: "green" }}>{textStatus}</Text> : null}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}
