import { Stack } from "expo-router";
import * as SplashScreen from "expo-splash-screen";
import { useFonts } from "expo-font";
import React from "react";
import { Text, TextInput } from "react-native";
import { AuthProvider } from "../src/auth/AuthContext";

const APP_FONT_FAMILY = "Exo2-Regular";
let defaultsApplied = false;

SplashScreen.preventAutoHideAsync().catch(() => undefined);

export default function RootLayout() {
  const [fontsLoaded] = useFonts({
    "Exo2-Regular": require("../assets/images/Exo_2/static/Exo2-Regular.ttf"),
    "Exo2-Medium": require("../assets/images/Exo_2/static/Exo2-Medium.ttf"),
    "Exo2-SemiBold": require("../assets/images/Exo_2/static/Exo2-SemiBold.ttf"),
    "Exo2-Bold": require("../assets/images/Exo_2/static/Exo2-Bold.ttf"),
    "Exo2-Italic": require("../assets/images/Exo_2/static/Exo2-Italic.ttf"),
  });

  React.useEffect(() => {
    if (!fontsLoaded) return;
    if (!defaultsApplied) {
      // Apply a single app-wide font so screens using inline styles also inherit it.
      Text.defaultProps = Text.defaultProps ?? {};
      Text.defaultProps.style = [{ fontFamily: APP_FONT_FAMILY }, Text.defaultProps.style];

      TextInput.defaultProps = TextInput.defaultProps ?? {};
      TextInput.defaultProps.style = [{ fontFamily: APP_FONT_FAMILY }, TextInput.defaultProps.style];
      defaultsApplied = true;
    }
    SplashScreen.hideAsync().catch(() => undefined);
  }, [fontsLoaded]);

  if (!fontsLoaded) {
    return null;
  }

  return (
    <AuthProvider>
      <Stack screenOptions={{ headerShown: false }}>
        <Stack.Screen name="auth" />
        <Stack.Screen name="(tabs)" />
      </Stack>
    </AuthProvider>
  );
}
