import React, { useState } from "react";
import { View, Text, TextInput, Pressable } from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import { Redirect } from "expo-router";

import { useAuth } from "../src/auth/AuthContext";

const FONT_SEMIBOLD = "Exo2-SemiBold";

export default function AuthScreen() {
  const { isAuthenticated, isHydrated, login, register } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (!isHydrated) {
    return null;
  }

  if (isAuthenticated) {
    return <Redirect href="/timeline" />;
  }

  async function submit() {
    if (!username.trim() || !password.trim()) {
      setError("Username and password are required.");
      return;
    }
    setBusy(true);
    setError(null);
    try {
      if (mode === "login") {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password, name.trim() || undefined);
      }
    } catch (err: any) {
      setError(err?.message ?? "Authentication failed.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <SafeAreaView style={{ flex: 1, backgroundColor: "#F7F8FC" }}>
      <View style={{ flex: 1, justifyContent: "center", padding: 22, gap: 12 }}>
        <Text style={{ fontSize: 30, fontFamily: FONT_SEMIBOLD, color: "#1D2433" }}>Vital</Text>
        <Text style={{ color: "#5C6784" }}>
          {mode === "login" ? "Sign in to continue." : "Create your account."}
        </Text>

        {mode === "register" ? (
          <TextInput
            value={name}
            onChangeText={setName}
            placeholder="Preferred first name (optional)"
            style={{ borderWidth: 1, borderColor: "#D6DCEB", borderRadius: 10, padding: 12, backgroundColor: "#FFF" }}
          />
        ) : null}
        <TextInput
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
          placeholder="Username"
          style={{ borderWidth: 1, borderColor: "#D6DCEB", borderRadius: 10, padding: 12, backgroundColor: "#FFF" }}
        />
        <TextInput
          value={password}
          onChangeText={setPassword}
          secureTextEntry
          placeholder="Password"
          style={{ borderWidth: 1, borderColor: "#D6DCEB", borderRadius: 10, padding: 12, backgroundColor: "#FFF" }}
        />

        <Pressable
          onPress={submit}
          disabled={busy}
          style={{
            backgroundColor: "#2E5BCE",
            borderRadius: 10,
            paddingVertical: 12,
            alignItems: "center",
            opacity: busy ? 0.7 : 1,
          }}
        >
          <Text style={{ color: "#FFF", fontFamily: FONT_SEMIBOLD }}>
            {busy ? "Please wait..." : mode === "login" ? "Sign In" : "Create Account"}
          </Text>
        </Pressable>

        <Pressable onPress={() => setMode(mode === "login" ? "register" : "login")} style={{ alignItems: "center", padding: 8 }}>
          <Text style={{ color: "#2E5BCE" }}>
            {mode === "login" ? "Need an account? Register" : "Have an account? Sign in"}
          </Text>
        </Pressable>

        {error ? <Text style={{ color: "#B42318" }}>{error}</Text> : null}
      </View>
    </SafeAreaView>
  );
}
