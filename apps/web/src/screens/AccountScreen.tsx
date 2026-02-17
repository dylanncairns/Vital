import React, { useMemo, useState } from "react";
import { Alert, Pressable, Text, TextInput, View } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

import { useAuth } from "../auth/AuthContext";

const FONT_SEMIBOLD = "Exo2-SemiBold";

export default function AccountScreen() {
  const { user, logout, updateName, deleteAccount } = useAuth();
  const [name, setName] = useState(user?.name ?? "");
  const [editingName, setEditingName] = useState(false);
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const username = useMemo(() => user?.username ?? "", [user]);

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

  async function saveName() {
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      await updateName(name.trim());
      setStatus("Preferred first name updated.");
      setEditingName(false);
    } catch (err: any) {
      setError(err?.message ?? "Failed to update name.");
    } finally {
      setBusy(false);
    }
  }

  async function onDeleteAccountPress() {
    if (busy) return;
    const confirmed = await confirmAction(
      "Delete account?",
      "This will permanently remove your account and timeline data."
    );
    if (!confirmed) return;
    setBusy(true);
    setError(null);
    setStatus(null);
    try {
      await deleteAccount();
    } catch {
      // Keep delete failures silent in UI per product preference.
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#F7F8FC" }}>
      <View style={{ padding: 16, gap: 12 }}>
        <View style={{ paddingTop: 14, paddingBottom: 10, alignItems: "center", justifyContent: "center", minHeight: 44 }}>
          <Text style={{ fontSize: 28, fontFamily: "Exo2-Bold", color: "#101426", textAlign: "center" }}>Account</Text>
        </View>
        <Text style={{ color: "#5C6784" }}>Username: {username}</Text>
        <Text style={{ color: "#5C6784" }}>Preferred first name: {user?.name ?? "-"}</Text>

        {!editingName ? (
          <Pressable
            onPress={() => {
              setName(user?.name ?? "");
              setStatus(null);
              setError(null);
              setEditingName(true);
            }}
            style={{
              backgroundColor: "#2E5BCE",
              borderRadius: 10,
              paddingVertical: 12,
              alignItems: "center",
            }}
          >
            <Text style={{ color: "#FFF", fontFamily: FONT_SEMIBOLD }}>Change Preferred First Name</Text>
          </Pressable>
        ) : (
          <View style={{ gap: 10 }}>
            <TextInput
              value={name}
              onChangeText={setName}
              placeholder="Preferred first name"
              style={{ borderWidth: 1, borderColor: "#D6DCEB", borderRadius: 10, padding: 12, backgroundColor: "#FFF" }}
            />
            <View style={{ flexDirection: "row", gap: 8 }}>
              <Pressable
                onPress={saveName}
                disabled={busy}
                style={{
                  flex: 1,
                  backgroundColor: "#2E5BCE",
                  borderRadius: 10,
                  paddingVertical: 12,
                  alignItems: "center",
                  opacity: busy ? 0.7 : 1,
                }}
              >
                <Text style={{ color: "#FFF", fontFamily: FONT_SEMIBOLD }}>{busy ? "Saving..." : "Save First Name"}</Text>
              </Pressable>
              <Pressable
                onPress={() => {
                  setEditingName(false);
                  setName(user?.name ?? "");
                }}
                style={{
                  flex: 1,
                  borderWidth: 1,
                  borderColor: "#D0D5DD",
                  borderRadius: 10,
                  paddingVertical: 12,
                  alignItems: "center",
                  backgroundColor: "#FFFFFF",
                }}
              >
                <Text style={{ color: "#1D2433", fontFamily: FONT_SEMIBOLD }}>Cancel</Text>
              </Pressable>
            </View>
          </View>
        )}

        <Pressable
          onPress={logout}
          style={{
            borderWidth: 1,
            borderColor: "#D0D5DD",
            borderRadius: 10,
            paddingVertical: 12,
            alignItems: "center",
            backgroundColor: "#FFFFFF",
          }}
        >
          <Text style={{ color: "#1D2433", fontFamily: FONT_SEMIBOLD }}>Logout</Text>
        </Pressable>

        {status ? <Text style={{ color: "#0A7A4F" }}>{status}</Text> : null}
        {error ? <Text style={{ color: "#B42318" }}>{error}</Text> : null}
        <Pressable
          onPress={onDeleteAccountPress}
          disabled={busy}
          style={{
            marginTop: 6,
            alignItems: "center",
            opacity: busy ? 0.6 : 1,
          }}
        >
          <Text style={{ color: "#8C92A6", fontFamily: "Exo2-Regular", fontSize: 12 }}>Delete account</Text>
        </Pressable>
      </View>
    </SafeAreaView>
  );
}
