import { Tabs } from "expo-router";
import React from "react";

// Navigation tabs for base screens
// Base screens include timeline display and event logging interface in this git
export default function TabLayout() {
  return (
    <Tabs screenOptions={{ headerShown: false }}>
      <Tabs.Screen name="index" options={{ href: null }} />
      <Tabs.Screen name="timeline" options={{ title: "Timeline" }} />
      <Tabs.Screen name="logevent" options={{ title: "Log Event" }} />
    </Tabs>
  );
}
