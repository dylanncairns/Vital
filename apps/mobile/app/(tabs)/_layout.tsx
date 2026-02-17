import { Redirect, Tabs } from "expo-router";
import React from "react";
import { useAuth } from "../../src/auth/AuthContext";
const APP_FONT_FAMILY = "Exo2-SemiBold";

// Navigation tabs for base screens
// Base screens include timeline display and event logging interface in this git
export default function TabLayout() {
  const { isAuthenticated, isHydrated } = useAuth();
  if (!isHydrated) {
    return null;
  }
  if (!isAuthenticated) {
    return <Redirect href="/auth" />;
  }
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: "#FFFFFF",
          borderTopColor: "#E2E7F1",
          borderTopWidth: 1,
        },
        tabBarLabelStyle: { fontFamily: APP_FONT_FAMILY, color: "#343A52" },
        tabBarActiveTintColor: "#343A52",
        tabBarInactiveTintColor: "#343A52",
      }}
    >
      <Tabs.Screen name="index" options={{ href: null }} />
      <Tabs.Screen name="timeline" options={{ title: "Timeline" }} />
      <Tabs.Screen name="insights" options={{ title: "Insights" }} />
      <Tabs.Screen name="logevent" options={{ title: "Log Event" }} />
      <Tabs.Screen name="account" options={{ title: "Account" }} />
    </Tabs>
  );
}
